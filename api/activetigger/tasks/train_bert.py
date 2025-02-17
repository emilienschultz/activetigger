import gc
import json
import logging
import multiprocessing
import os
import shutil
from pathlib import Path
from typing import Optional

import datasets
import torch
from pandas import DataFrame
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from activetigger.tasks.base_task import BaseTask, CustomLoggingCallback


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
    """

    def __init__(
        self,
        path: Path,
        name: str,
        df: DataFrame,
        col_text: str,
        col_label: str,
        base_model: str,
        params: dict,
        test_size: float,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        self.path = path
        self.name = name
        self.df = df
        self.col_text = col_text
        self.col_label = col_label
        self.base_model = base_model
        self.params = params
        self.test_size = test_size
        self.event = event
        self.unique_id = unique_id

    def process(self):
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
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
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
        training_data = self.df[[self.col_label, self.col_label]]
        self.df["labels"] = self.df[self.col_label].copy().replace(label2id)
        self.df["text"] = self.df[self.col_text]
        self.df = datasets.Dataset.from_pandas(self.df[["text", "labels"]])

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if self.params["adapt"]:
            self.df = self.df.map(
                lambda e: tokenizer(
                    e["text"],
                    truncation=True,
                    padding=True,
                    max_length=512,
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
                    max_length=512,
                    return_tensors="pt",
                ),
                batched=True,
            )

        # Build train/test dataset for dev eval
        self.df = self.df.train_test_split(
            test_size=self.test_size
        )  # stratify_by_column="label"
        logger.info("Train/test dataset created")

        # Model
        bert = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        logger.info("Model loaded")
        print("Model loaded")

        try:
            total_steps = (float(self.params["epochs"]) * len(self.df["train"])) // (
                int(self.params["batchsize"]) * float(self.params["gradacc"])
            )
            warmup_steps = int((total_steps) // 10)
            eval_steps = total_steps // self.params["eval"]
            training_args = TrainingArguments(
                output_dir=current_path.joinpath("train"),
                logging_dir=current_path.joinpath("logs"),
                learning_rate=float(self.params["lrate"]),
                weight_decay=float(self.params["wdecay"]),
                num_train_epochs=float(self.params["epochs"]),
                gradient_accumulation_steps=int(self.params["gradacc"]),
                per_device_train_batch_size=int(self.params["batchsize"]),
                per_device_eval_batch_size=int(self.params["batchsize"]),
                warmup_steps=int(warmup_steps),
                eval_steps=eval_steps,
                evaluation_strategy="steps",
                save_strategy="steps",
                save_steps=int(eval_steps),
                logging_steps=int(eval_steps),
                do_eval=True,
                greater_is_better=False,
                load_best_model_at_end=self.params["best"],
                metric_for_best_model="eval_loss",
                use_cpu=not bool(self.params["gpu"]),  # deactivate gpu
                # optim="adamw_torch_fused", # improved optimizer
            )

            # Train
            trainer = Trainer(
                model=bert,
                args=training_args,
                train_dataset=self.df["train"],
                eval_dataset=self.df["test"],
                callbacks=[
                    CustomLoggingCallback(
                        self.event, current_path=current_path, logger=logger
                    )
                ],
            )
            trainer.train()

            # save model
            bert.save_pretrained(current_path)
            logger.info(f"Model trained {current_path}")

            # save training data in a file
            training_data.to_parquet(current_path.joinpath("training_data.parquet"))

            # save parameters in a file
            self.params["test_size"] = self.test_size
            self.params["base_model"] = self.base_model
            with open(current_path.joinpath("parameters.json"), "w") as f:
                json.dump(self.params, f)

            # remove intermediate steps and logs if succeed
            shutil.rmtree(current_path.joinpath("train"))
            os.rename(log_path, current_path.joinpath("finished"))

            # save log history of the training for statistics
            with open(current_path.joinpath("log_history.txt"), "w") as f:
                json.dump(trainer.state.log_history, f)

            return {"success": "Model trained"}

        except Exception as e:
            print("Error in training", e)
            shutil.rmtree(current_path)
        finally:
            del trainer, bert, self.df, device, self.event
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    def clean(self, result):
        if not isinstance(result, dict):
            raise Exception("Probleme with the function")
        pass

        if "path" in result and "predict_train.parquet" in result["path"]:
            return result["path"]
