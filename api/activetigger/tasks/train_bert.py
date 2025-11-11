import gc
import json
import logging
import multiprocessing
import os
import shutil
from collections import Counter
from logging import Logger
from pathlib import Path
from typing import Optional

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
from activetigger.datamodels import LMParametersModel
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask

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

    def __call__(self) -> None:
        """
        Main process to the task
        """
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

        #  create repertory for the specific model
        current_path = self.path.joinpath(self.name)
        if not current_path.exists():
            os.makedirs(current_path)

        # logging the process
        log_path = current_path.joinpath("status.log")
        logger = logging.getLogger("train_bert_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {self.base_model}")

        # test labels missing values and remove them
        if self.df[self.col_label].isnull().sum() > 0:
            self.df = self.df[self.df[self.col_label].notnull()]
            logger.info(f"Missing labels - reducing training data to {len(self.df)}")

        # test empty texts and remove them
        if self.df[self.col_text].isnull().sum() > 0:
            self.df = self.df[self.df[self.col_text].notnull()]
            logger.info(f"Missing texts - reducing training data to {len(self.df)}")

        # formatting data
        # alphabetical order
        labels = sorted(list(self.df[self.col_label].dropna().unique()))
        label2id = {j: i for i, j in enumerate(labels)}
        id2label = {i: j for i, j in enumerate(labels)}
        training_data = self.df[[self.col_text, self.col_label]]
        self.df["labels"] = self.df[self.col_label].copy().replace(label2id)
        self.df["text"] = self.df[self.col_text]
        self.df = datasets.Dataset.from_pandas(self.df[["text", "labels"]])

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model, trust_remote_code=True)
        if self.params.adapt:
            self.df = self.df.map(
                lambda e: tokenizer(
                    e["text"],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                ),
                batched=True,
            )
        else:
            self.df = self.df.map(
                lambda e: tokenizer(
                    e["text"],
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                ),
                batched=True,
            )

        # Build train/test dataset for dev eval
        self.df = self.df.train_test_split(test_size=self.test_size)  # stratify_by_column="label"
        logger.info("Train/test dataset created")

        # Model
        bert = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        )

        logger.info("Model loaded")
        print("Model loaded")

        try:
            total_steps = (float(self.params.epochs) * len(self.df["train"])) // (
                int(self.params.batchsize) * float(self.params.gradacc)
            )
            warmup_steps = int((total_steps) // 10)
            eval_steps = (total_steps - warmup_steps) // self.params.eval
            if eval_steps == 0:
                eval_steps = 1

            training_args = TrainingArguments(
                output_dir=str(current_path.joinpath("train")),
                logging_dir=str(current_path.joinpath("logs")),
                learning_rate=float(self.params.lrate),
                weight_decay=float(self.params.wdecay),
                num_train_epochs=float(self.params.epochs),
                gradient_accumulation_steps=int(self.params.gradacc),
                per_device_train_batch_size=int(self.params.batchsize),
                per_device_eval_batch_size=int(self.params.batchsize),
                warmup_steps=int(warmup_steps),
                eval_strategy="steps",
                eval_steps=eval_steps,
                save_strategy="best",  # steps
                save_steps=int(eval_steps),
                logging_steps=int(eval_steps),
                do_eval=True,
                greater_is_better=False,
                load_best_model_at_end=self.params.best,
                metric_for_best_model="eval_loss",
                use_cpu=not bool(self.params.gpu),  # deactivate gpu
            )

            if self.loss == "cross_entropy":
                trainer = Trainer(
                    model=bert,
                    args=training_args,
                    train_dataset=self.df["train"],
                    eval_dataset=self.df["test"],
                    callbacks=[
                        CustomLoggingCallback(self.event, current_path=current_path, logger=logger)
                    ],
                )
            elif self.loss == "weighted_cross_entropy":
                print("Using weighted cross entropy loss - EXPERIMENTAL")
                trainer = CustomTrainer(
                    model=bert,
                    args=training_args,
                    train_dataset=self.df["train"],
                    eval_dataset=self.df["test"],
                    callbacks=[
                        CustomLoggingCallback(self.event, current_path=current_path, logger=logger)
                    ],
                    class_weights=compute_class_weights(self.df["train"], label_key="labels"),
                )
            else:
                raise ValueError(f"Loss function {self.loss} not recognized.")

            trainer.train()  # type: ignore[attr-defined]

            # predict on the data (separation validation set and training set)
            predictions_test = trainer.predict(self.df["test"])  # type: ignore[attr-defined]
            predictions_train = trainer.predict(self.df["train"])  # type: ignore[attr-defined]

            # shape and write the data of the test set
            test = self.df["test"].to_pandas().set_index("id")
            test["true_label"] = [id2label[i] for i in predictions_test.label_ids]
            test["predicted_label"] = [
                id2label[i] for i in np.argmax(predictions_test.predictions, axis=1)
            ]
            test[["true_label", "predicted_label"]].to_csv(
                current_path.joinpath("test_dataset_eval.csv")
            )

            # shape and write the data of the train set
            train = self.df["train"].to_pandas().set_index("id")
            train["true_label"] = [id2label[i] for i in predictions_train.label_ids]
            train["predicted_label"] = [
                id2label[i] for i in np.argmax(predictions_train.predictions, axis=1)
            ]
            train[["true_label", "predicted_label"]].to_csv(
                current_path.joinpath("train_dataset_eval.csv")
            )

            # save model
            bert.save_pretrained(current_path)
            logger.info(f"Model trained {current_path}")

            # save training data in a file
            training_data.to_parquet(current_path.joinpath("training_data.parquet"))

            # save parameters of the trained model in a file
            params_to_save = self.params.model_dump()
            params_to_save["test_size"] = self.test_size
            params_to_save["base_model"] = self.base_model
            params_to_save["n_train"] = len(self.df["train"])

            with open(current_path.joinpath("parameters.json"), "w") as f:
                json.dump(params_to_save, f)

            # compute metrics and write
            metrics_train = get_metrics(
                train["true_label"], train["predicted_label"], train["text"]
            )
            metrics_test = get_metrics(test["true_label"], test["predicted_label"], test["text"])
            with open(str(current_path.joinpath("metrics_training.json")), "w") as f:
                json.dump(
                    {
                        "train": metrics_train.model_dump(mode="json"),
                        "trainvalid": metrics_test.model_dump(mode="json"),
                    },
                    f,
                )

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

            return None

        except Exception as e:
            print("Error in training", e)
            shutil.rmtree(current_path)
            raise e
        finally:
            print("Cleaning memory")
            try:
                del (
                    trainer,
                    bert,
                    self.df,
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
