import gc
import json
import logging
import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from transformers import (  # type: ignore[import]
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from activetigger.datamodels import MLStatisticsModel, ReturnTaskPredictModel
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class PredictBert(BaseTask):
    """
    Class to predict with a bert model

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str, Optional): label column
    base_model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution
    event : possibility to interrupt
    unique_id : unique id for the current task

    if statistic & col_label, compute specific statistics
    """

    kind = "predict_bert"

    def __init__(
        self,
        path: Path,
        df: DataFrame,
        col_text: str,
        col_label: str | None = None,
        col_id: str | None = None,
        col_datasets: str | None = None,
        file_name: str = "predict.parquet",
        batch: int = 32,
        statistics: list | None = None,
        event: Optional[multiprocessing.synchronize.Event] = None,
        unique_id: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.df = df
        self.col_text = col_text
        self.col_label = col_label
        self.col_id = col_id
        self.col_datasets = col_datasets
        self.event = event
        self.unique_id = unique_id
        self.file_name = file_name
        self.batch = batch
        self.statistics = statistics

        if col_text not in self.df.columns:
            raise ValueError(f"Column text {col_text} not in dataframe")

        if col_label is not None and col_label not in self.df.columns:
            raise ValueError(f"Column label {col_label} not in dataframe")

        if col_datasets is not None and col_datasets not in self.df.columns:
            raise ValueError(f"Column datasets {col_datasets} not in dataframe")

        if col_id is not None and col_id not in self.df.columns:
            raise ValueError(f"Column id {col_id} not in dataframe")

        if statistics is not None and col_label is None:
            raise ValueError("Column label must be provided to compute statistics")

    def __call__(self) -> ReturnTaskPredictModel:
        """
        Main process to predict
        """
        print("start predicting")

        # empty cache
        torch.cuda.empty_cache()

        # check if GPU available
        gpu = False
        if torch.cuda.is_available():
            print("GPU is available")
            gpu = True

        # logging the process
        log_path = self.path / "status_predict.log"
        progress_path = self.path / "progress_predict"
        logger = logging.getLogger("predict_bert_model")
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        print("load model")
        with open(self.path / "parameters.json", "r") as jsonfile:
            data = json.load(jsonfile)
            max_length = data.get("max_length", 512)
            if "base_model" in data:
                modeltype = data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")
        tokenizer = AutoTokenizer.from_pretrained(modeltype, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.path, trust_remote_code=True
        )

        print("function prediction : start")
        if torch.cuda.is_available():
            model.cuda()

        try:
            # Start prediction with batches
            predictions = []
            # logging the process
            for chunk in [
                self.df[self.col_text][i : i + self.batch]
                for i in range(0, self.df.shape[0], self.batch)
            ]:
                # user interrupt
                if self.event.is_set():
                    logger.info("Event set, stopping training.")
                    raise Exception("Event set, stopping training.")

                print("Next chunck prediction")
                chunk = tokenizer(
                    list(chunk),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=max_length,
                )
                if gpu:
                    chunk = chunk.to("cuda")
                with torch.no_grad():
                    outputs = model(**chunk)
                res = outputs[0]
                if gpu:
                    res = res.cpu()
                res = res.softmax(1).detach().numpy()
                predictions.append(res)

                # write progress
                with open(progress_path, "w") as f:
                    f.write(str((len(predictions) * self.batch / self.df.shape[0]) * 100))

            # to dataframe
            pred = pd.DataFrame(
                np.concatenate(predictions),
                columns=sorted(list(model.config.label2id.keys())),
                index=self.df.index,
            )

            entropy = -1 * (pred * np.log(pred)).sum(axis=1)
            pred["entropy"] = entropy
            pred["prediction"] = pred.drop(columns="entropy").idxmax(axis=1)

            # add columns if available
            if self.col_datasets:
                pred[self.col_datasets] = self.df[self.col_datasets]
            if self.col_id:
                pred[self.col_id] = self.df[self.col_id]
            if self.col_label:
                pred["label"] = self.df[self.col_label]

            print("Prediction ended")

            # save the file of predictions
            pred.to_parquet(self.path.joinpath(self.file_name))

            # return if no statistics
            if self.statistics is None:
                return ReturnTaskPredictModel(
                    path=str(self.path.joinpath(self.file_name)), metrics=None
                )

            # case where statistics should be computed

            print("Compute statistics")
            metrics: dict[str, MLStatisticsModel] = {}

            # add text in the dataframe to be able to get mismatch
            pred["text"] = self.df[self.col_text]
            filter_label = pred["label"].notna()  # only non null values

            # compute the statistics per dataset
            for dataset in self.statistics:
                filter_dataset = pred[self.col_datasets] == dataset
                filter = filter_label & filter_dataset
                if filter.sum() < 5:
                    continue
                metrics[dataset] = get_metrics(
                    pred[filter]["label"],
                    pred[filter]["prediction"],
                    pred["text"],
                )

            # add out of sample (labelled data not in training data)
            index_model = pd.read_parquet(
                self.path.joinpath("training_data.parquet"), columns=[]
            ).index
            filter_oos = (
                ~pred.index.isin(index_model) & filter_label & pred[self.col_datasets] == "train"
            )
            if filter_oos.sum() > 10:
                metrics["outofsample"] = get_metrics(
                    pred[filter_oos]["label"], pred[filter_oos]["prediction"], pred["text"]
                )

            # write the metrics in a json file
            with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
                json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)

            # drop the temporary text col
            pred.drop(columns=["text"], inplace=True)

            return ReturnTaskPredictModel(
                path=str(self.path.joinpath(self.file_name)), metrics=metrics
            )
        except Exception as e:
            print("Error in prediction", e)
            raise e
        finally:
            # delete the logs
            os.remove(log_path)
            os.remove(progress_path)
            # clean memory
            del tokenizer, model, chunk, self.df, res, predictions, outputs, self.event
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
