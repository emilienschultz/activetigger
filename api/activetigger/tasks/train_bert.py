import gc
import json
import logging
import multiprocessing
import os
import shutil
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Any, Optional, Tuple

import datasets  # type: ignore[import]  # type: ignore[import]
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import nn
from transformers import (  # type: ignore[import]  # type: ignore[import]
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from activetigger.config import config
from activetigger.datamodels import EventsModel, LMParametersModel, MLStatisticsModel
from activetigger.functions import get_metrics
from activetigger.monitoring import TaskTimer
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.utils import length_after_tokenizing, retrieve_model_max_length

pd.set_option("future.no_silent_downcasting", True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomLoggingCallback(TrainerCallback):
    event: Optional[multiprocessing.synchronize.Event]
    current_path: Path
    logger: Logger

    def __init__(self, event, logger, current_path):
        super().__init__()
        self.event = event
        self.current_path = current_path
        self.logger = logger

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.logger.info(f"Step {state.global_step}")
        progress_percentage = (state.global_step / state.max_steps) * 100
        with open(self.current_path.joinpath("progress_train"), "w") as f:
            f.write(str(progress_percentage))
        with open(self.current_path.joinpath("log_history.txt"), "w") as f:
            json.dump(state.log_history, f)
        # end if event set
        if self.event is not None:
            if self.event.is_set():
                self.logger.info("Event set, stopping training.")
                control.should_training_stop = True
                raise Exception("Process interrupted by user")


# Function for the weighted loss computation


# Rescaling the weights
def compute_class_weights(dataset, label_key="labels"):
    labels = [example[label_key] for example in dataset]
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    # Inverse frequency weight
    weights = [total / (num_classes * label_counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float)


# CustomTrainer is a subclass of Trainer that allows for custom loss computation.
# https://stackoverflow.com/questions/70979844/using-weights-with-transformers-huggingface
class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        print("CustomTrainer initialized with class weights:", self.class_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Use dynamic weights
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


class TrainBert(BaseTask):
    """
    Class to train a bert model

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    base_model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution
    event : possibility to interrupt
    unique_id : unique id for the current task
    loss : loss function to use (cross_entropy, weighted_cross_entropy)

    TODO : test more weighted loss entropy
    """

    kind = "train_bert"

    def __init__(
        self,
        path: Path,
        project_slug: str,
        model_name: str,
        df: DataFrame | datasets.Dataset,
        col_text: str,
        col_label: str,
        base_model: str,
        params: LMParametersModel,
        test_size: float,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        loss: Optional[str] = "cross_entropy",
        max_length: int = 512,
        auto_max_length: bool = False,
        **kwargs,
    ):
        self.path = path
        self.project_slug = project_slug
        self.name = model_name
        df.index.name = "id"
        self.df = df
        self.col_text = col_text
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id
        self.loss = loss
        self.max_length = max_length
        self.auto_max_length = auto_max_length

    def __init_paths(self) -> Tuple[Path, Path]:
        """Initiate the current path (directory for the model) and for the logger"""
        #  create repertory for the specific model
        current_path = self.path.joinpath(self.name)
        if not current_path.exists():
            os.makedirs(current_path)
        # logging the process
        log_path = current_path.joinpath("status.log")
        return current_path, log_path

    def __init_logger(self, log_path) -> Logger:
        """Load the logger and set it up"""
        logger = logging.getLogger("train_bert_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {self.base_model}")
        return logger

    def __init_device(self) -> torch.device:
        """Choose the device, first try to use cuda, then mps and finally cpu"""
        # Pick up the type of memory to use
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            device = torch.device("cuda")
            print("Using CUDA for computation")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS for computation")
        else:
            device = torch.device("cpu")
            print("Using CPU for computation")
        return device

    def __check_data(self, df: pd.DataFrame, col_label: str, col_text: str) -> pd.DataFrame:
        """Remove rows missing labels or text"""
        df = df.copy()
        # test labels missing values and remove them
        if df[col_label].isnull().sum() > 0:
            df = df[df[col_label].notnull()]
            self.logger.info(f"Missing labels - reducing training data to {len(df)}")

        # test empty texts and remove them
        if df[col_text].isnull().sum() > 0:
            df = df[df[col_text].notnull()]
            self.logger.info(f"Missing texts - reducing training data to {len(df)}")
        return df

    def __retrieve_labels(self, df: pd.DataFrame, col_label):
        # formatting data
        # alphabetical order
        labels = sorted(list(df[col_label].dropna().unique()))

        if len(labels) < 2:
            raise ValueError(
                "Not enough classes. Either you excluded classes or there are not enough annotations."
            )

        label2id = {j: i for i, j in enumerate(labels)}
        id2label = {i: j for i, j in enumerate(labels)}
        return labels, label2id, id2label

    def __transform_to_dataset(
        self, df: pd.DataFrame, col_label: str, col_text: str, label2id: dict[str, int]
    ) -> datasets.Dataset:
        """Transform the dataframe into a dataset with the right format for
        training"""
        df = df.copy()
        df["text"] = df[col_text]
        df["labels"] = df[col_label].copy().replace(label2id)
        return datasets.Dataset.from_pandas(df[["text", "labels"]])

    def __load_tokenizer(self, base_model: str):
        """Load the tokenize"""
        return AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    def __cap_tokenizer_max_length(
        self,
        texts: pd.Series,
        tokenizer,
        auto_max_length: bool,
        original_max_length: int,
        base_model_max_length: int,
        adapt: bool,
    ) -> Tuple[Any, int]:
        """Cap the tokenizer max length and create a tokenizing function"""
        # if auto_max_length set max_length to the maximum length of tokenized sentences
        # Tokenize the text column
        get_n_tokens = lambda txt: length_after_tokenizing(txt, tokenizer)
        if auto_max_length:
            max_length = int(texts.apply(get_n_tokens).dropna().max())

        # cap max_length
        max_length = min(original_max_length, base_model_max_length)
        # evaluate the proportion of elements truncated
        percentage_truncated = int(
            100 * texts.apply(get_n_tokens).dropna().apply(lambda x: x > max_length).mean()
        )

        if adapt:
            tokenizing_function = lambda e: tokenizer(
                e["text"],
                truncation=True,
                padding=True,
                return_tensors="pt",
                max_length=int(self.max_length),
            )
        else:
            tokenizing_function = lambda e: tokenizer(
                e["text"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                max_length=self.max_length,
            )
        return tokenizing_function, percentage_truncated

    def __load_trainer(
        self,
        current_path: Path,
        ds: datasets.DatasetDict,
        bert_model,
        params: LMParametersModel,
        loss: str,
    ) -> Trainer:
        """Load the training arguments and update the configuration"""

        # Calculate the number of steps (total, warmup and eval)
        total_steps = (float(params.epochs) * len(ds["train"])) // (
            int(params.batchsize) * float(params.gradacc)
        )
        warmup_steps = int((total_steps) // 10)
        eval_steps = (total_steps - warmup_steps) // params.eval
        if eval_steps == 0:
            eval_steps = 1

        # Load the training arguments
        training_args = TrainingArguments(
            # Directories
            output_dir=str(current_path.joinpath("train")),
            logging_dir=str(current_path.joinpath("logs")),
            # Hyperparameters
            learning_rate=float(params.lrate),
            weight_decay=float(params.wdecay),
            num_train_epochs=float(params.epochs),
            warmup_steps=int(warmup_steps),
            # Batch sizes
            gradient_accumulation_steps=int(params.gradacc),
            per_device_train_batch_size=int(params.batchsize),
            per_device_eval_batch_size=int(params.batchsize),
            # Logging and saving parameters
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="best",  # steps
            metric_for_best_model="eval_loss",
            save_steps=int(eval_steps),
            logging_steps=int(eval_steps),
            do_eval=True,
            greater_is_better=False,
            load_best_model_at_end=params.best,
            use_cpu=not bool(params.gpu),  # deactivate gpu
        )

        callback = CustomLoggingCallback(self.event, current_path=current_path, logger=self.logger)
        if loss == "cross_entropy":
            trainer = Trainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"],
                callbacks=[callback],
            )
        elif loss == "weighted_cross_entropy":
            print("Using weighted cross entropy loss - EXPERIMENTAL")
            trainer = CustomTrainer(
                model=bert_model,
                args=training_args,
                train_dataset=ds["train"],
                eval_dataset=ds["test"],
                callbacks=[callback],
                class_weights=compute_class_weights(ds["train"], label_key="labels"),
            )
        else:
            raise ValueError(f"Loss function {loss} not recognized.")

        return trainer

    def __create_save_files(
        self,
        current_path: Path,
        log_path: Path,
        df_train_results: pd.DataFrame,
        df_test_results: pd.DataFrame,
        training_data: pd.DataFrame,
        bert_model,
        params_to_save: dict[str, Any],
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel,
    ) -> None:
        """Save the model and parameters
        Save the following objects:
        - predictions of the train set (csv)
        - predictions of the test set  (csv)
        - data used during the training (parquet)
        - the trained model
        - the parameters used during the training (json)
        - metrics (json)

        Also delete intermediate files
        """

        # Save results for the train and test set
        df_train_results[["true_label", "predicted_label"]].to_csv(
            current_path.joinpath("train_dataset_eval.csv")
        )
        df_test_results[["true_label", "predicted_label"]].to_csv(
            current_path.joinpath("test_dataset_eval.csv")
        )
        training_data.to_parquet(current_path.joinpath("training_data.parquet"))

        # save the trained bert model
        bert_model.save_pretrained(current_path)

        # Save parameters
        with open(current_path.joinpath("parameters.json"), "w") as f:
            json.dump(params_to_save, f)

        # remove intermediate steps and logs if succeed
        shutil.rmtree(current_path.joinpath("train"))
        os.rename(log_path, current_path.joinpath("finished"))

        # make archive (create dir if needed)
        path_static = f"{config.data_path}/projects/static/{self.project_slug}"
        os.makedirs(path_static, exist_ok=True)
        shutil.make_archive(
            f"{path_static}/{self.name}",
            "gztar",
            str(self.path.joinpath(self.name)),
        )

        with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
            json.dump(
                {
                    "train": metrics_train.model_dump(mode="json"),
                    "trainvalid": metrics_test.model_dump(mode="json"),
                },
                f,
            )

    def __call__(self) -> EventsModel:
        """
        Main process to the task
        """
        task_timer = TaskTimer(compulsory_steps=["setup", "train", "evaluate", "save_files"])
        task_timer.start("setup")

        current_path, log_path = self.__init_paths()
        self.logger = self.__init_logger(log_path)
        device = self.__init_device()

        self.df = self.__check_data(self.df, self.col_label, self.col_text)
        labels, label2id, id2label = self.__retrieve_labels(self.df, self.col_label)
        self.ds = self.__transform_to_dataset(self.df, self.col_label, self.col_text, label2id)

        tokenizer = self.__load_tokenizer(self.base_model)
        tokenizing_function, percentage_truncated = self.__cap_tokenizer_max_length(
            texts=self.df[self.col_text],
            tokenizer=tokenizer,
            auto_max_length=self.auto_max_length,
            original_max_length=self.max_length,
            base_model_max_length=retrieve_model_max_length(self.base_model),
            adapt=self.params.adapt,
        )
        self.ds = self.ds.map(tokenizing_function, batched=True)

        # Build train/test dataset for dev eval
        self.ds = self.ds.train_test_split(test_size=self.test_size)  # stratify_by_column="label"
        self.logger.info("Train/test dataset created")

        # Model
        bert_model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        )
        self.logger.info("Model loaded")
        print("Model loaded")

        try:
            trainer = self.__load_trainer(
                current_path, self.ds, bert_model, self.params, self.loss or "cross_entropy"
            )
            task_timer.stop("setup")

            task_timer.start("train")
            trainer.train()  # type: ignore[attr-defined]
            self.logger.info(f"Model trained {current_path}")
            task_timer.stop("train")

            # predict on the data (separation validation set and training set)
            task_timer.start("evaluate")
            predictions_test = trainer.predict(self.ds["test"])  # type: ignore[attr-defined]
            predictions_train = trainer.predict(self.ds["train"])  # type: ignore[attr-defined]

            # Compute the metrics
            df_train_results = self.ds["train"].to_pandas().set_index("id")
            df_train_results["true_label"] = [id2label[i] for i in predictions_train.label_ids]
            df_train_results["predicted_label"] = [
                id2label[i] for i in np.argmax(predictions_train.predictions, axis=1)
            ]
            metrics_train = get_metrics(
                Y_true=df_train_results["true_label"],
                Y_pred=df_train_results["predicted_label"],
                texts=df_train_results["text"],
                labels=labels,
            )
            df_test_results = self.ds["test"].to_pandas().set_index("id")
            df_test_results["true_label"] = [id2label[i] for i in predictions_test.label_ids]
            df_test_results["predicted_label"] = [
                id2label[i] for i in np.argmax(predictions_test.predictions, axis=1)
            ]
            metrics_test = get_metrics(
                Y_true=df_test_results["true_label"],
                Y_pred=df_test_results["predicted_label"],
                texts=df_test_results["text"],
                labels=labels,
            )
            task_timer.stop("evaluate")

            task_timer.start("save_files")
            params_to_save = self.params.model_dump()
            params_to_save.update(
                {
                    "test_size": self.test_size,
                    "base_model": self.base_model,
                    "n_train": len(self.ds["train"]),
                    "max_length": self.max_length,
                    "device": str(device),
                    "Proportion of elements truncated (%)": percentage_truncated,
                }
            )
            self.__create_save_files(
                current_path=current_path,
                log_path=log_path,
                df_train_results=df_train_results,
                df_test_results=df_test_results,
                training_data=self.df[[self.col_text, self.col_label]],
                bert_model=bert_model,
                params_to_save=params_to_save,
                metrics_train=metrics_train,
                metrics_test=metrics_test,
            )
            task_timer.stop("save_files")

        except Exception as e:
            print("Error in training", e)
            shutil.rmtree(current_path)
            raise e
        finally:
            print("Cleaning memory")
            try:
                del (
                    trainer,
                    bert_model,
                    self.df,
                    self.ds,
                    device,
                    self.event,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                gc.collect()

            except Exception as e:
                print("Error in cleaning memory", e)
                raise e

        return EventsModel(events=task_timer.get_events())
