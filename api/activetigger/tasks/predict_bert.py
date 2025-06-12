import gc
import json
import logging
import multiprocessing
import os
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

from activetigger.datamodels import ReturnTaskPredictModel
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
    statistics: Optional[str] = None

    def __init__(
        self,
        path: Path,
        df: DataFrame,
        col_text: str,
        col_label: str | None = None,
        col_id: str | None = None,
        batch: int = 32,
        file_name: str = "predict.parquet",
        statistics: str | None = None,
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
        self.event = event
        self.unique_id = unique_id
        self.file_name = file_name
        self.batch = batch

        # indicate if the statistics should be computed
        if statistics is not None and col_label is not None:
            self.statistics = statistics
        else:
            self.statistics = None

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
            if "base_model" in data:
                modeltype = data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")
        tokenizer = AutoTokenizer.from_pretrained(modeltype)
        model = AutoModelForSequenceClassification.from_pretrained(self.path)

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
                    max_length=512,
                    return_tensors="pt",
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

            # calculate entropy
            entropy = -1 * (pred * np.log(pred)).sum(axis=1)
            pred["entropy"] = entropy

            # calculate label
            pred["prediction"] = pred.drop(columns="entropy").idxmax(axis=1)
            print("Prediction ended")

            metrics = None

            # Optional, compute statistics
            if self.statistics:
                # add elements in the dataframe
                pred["label"] = self.df[self.col_label]
                pred["text"] = self.df[self.col_text]
                filter = pred["label"].notna()  # only non null values

                # case the statistics should be computed on all values
                if self.statistics == "full":
                    metrics = get_metrics(pred[filter]["label"], pred[filter]["prediction"])
                    # add full text disagreement
                    metrics.false_predictions = (
                        pred[["label", "prediction", "text"]]
                        .loc[list(metrics.false_predictions)]
                        .reset_index()
                    ).to_dict(orient="records")
                    # save in a dedicated file
                    with open(str(self.path.joinpath(f"metrics_{self.file_name}.json")), "w") as f:
                        json.dump(metrics.model_dump(mode="json"), f)

                # case the statistics should be computed only on element not in the training set
                # only if there is more than 10 elements
                if self.statistics == "outofsample":
                    index_model = pd.read_parquet(
                        self.path.joinpath("training_data.parquet"), columns=[]
                    ).index
                    filter_oos = ~pred.index.isin(index_model) & filter
                    if filter_oos.sum() > 10:
                        metrics = get_metrics(
                            pred[filter_oos]["label"], pred[filter_oos]["prediction"]
                        )
                        print(index_model)
                        metrics.false_predictions = (
                            pred[["label", "prediction", "text"]]
                            .loc[list(metrics.false_predictions)]
                            .reset_index()
                        ).to_dict(orient="records")
                        # save in a dedicated file
                        with open(str(self.path.joinpath("metrics_outofsample.json")), "w") as f:
                            json.dump(metrics.model_dump(mode="json"), f)
                    else:
                        print(
                            "Not enough out of sample data to compute statistics, only "
                            f"{filter_oos.sum()} elements"
                        )

                # drop the temporary text col
                pred.drop(columns=["text"], inplace=True)
                print("Add statistics")

            # add the column id original if available
            if self.col_id:
                pred[self.col_id] = self.df[self.col_id]

            # write the content in a parquet file
            pred.to_parquet(self.path.joinpath(self.file_name))
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
