import gc
import json
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

from activetigger.data import Data
from activetigger.datamodels import MLStatisticsModel, ReturnTaskPredictModel, TextDatasetModel
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class PredictBert(BaseTask):
    """
    Class to predict with a bert model
    """

    kind = "predict_bert"

    def __init__(
        self,
        dataset: str,
        path: Path,
        df: DataFrame | None,
        col_text: str,
        col_label: str | None = None,
        path_data: Path | None = None,
        external_dataset: TextDatasetModel | None = None,
        col_id_external: str | None = None,
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
        self.dataset = dataset
        self.col_text = col_text
        self.col_label = col_label
        self.col_id_external = col_id_external
        self.col_datasets = col_datasets
        self.event = event
        self.unique_id = unique_id
        self.file_name = file_name
        self.batch = batch
        self.statistics = statistics
        self.progress_path = self.path / "progress_predict"

        if self.df is None and path_data is not None:
            self.df = self.__load_external_file(path_data, external_dataset)

        if self.df is None:
            raise ValueError("Dataframe must be provided for prediction")

        if col_text not in self.df.columns:
            raise ValueError(f"Column text {col_text} not in dataframe")

        if col_label is not None and col_label not in self.df.columns:
            raise ValueError(f"Column label {col_label} not in dataframe")

        if col_datasets is not None and col_datasets not in self.df.columns:
            raise ValueError(f"Column datasets {col_datasets} not in dataframe")

        if col_id_external is not None and col_id_external not in self.df.columns:
            raise ValueError(f"Column id {col_id_external} not in dataframe")

        if statistics is not None and col_label is None:
            raise ValueError("Column label must be provided to compute statistics")

    def __load_external_file(
        self, path_data: Path, external_dataset: TextDatasetModel | None
    ) -> DataFrame:
        """
        Load file for prediction with specific rules to match the expected format
        """
        df = Data.read_dataset(path_data)

        if self.dataset == "external" and external_dataset is not None:
            df["text"] = df[external_dataset.text]
            df["index"] = df[external_dataset.id].apply(str)
            df["id_external"] = df["index"]
            df["dataset"] = "external"
            df.set_index("index", inplace=True)
            df = df[["id_external", "dataset", "text"]].dropna()

        if self.dataset == "all":
            df["id_external"] = df[self.col_id_external]
            df["dataset"] = "all"

        return df

    def __write_progress(self, progress: float) -> None:
        """
        Write progress to the progress file
        """
        with open(self.progress_path, "w") as f:
            f.write(str(progress))

    def __load_model(self, default_max_length: int = 512) -> tuple:
        """
        Load the model and tokenizer from the path
        """
        # read the config file
        with open(self.path / "parameters.json", "r") as jsonfile:
            data = json.load(jsonfile)
            max_length = int(data.get("max_length", default_max_length))
            if "base_model" in data:
                modeltype = data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")

        # load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(modeltype, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.path, trust_remote_code=True
        )

        return tokenizer, model, max_length

    def __listen_stop_event(self):
        """
        Raise exception if stop event is set
        """
        if self.event is not None:
            if self.event.is_set():
                raise Exception("Process interrupted by user")

    def __transform_to_dataframe(self, predictions: list, columns: list) -> DataFrame:
        """
        Transform the predictions to a dataframe
        """
        if self.df is None:
            raise ValueError("Dataframe is required to transform predictions")
        pred = pd.DataFrame(
            np.concatenate(predictions),
            columns=columns,
            index=self.df.index,
        )

        entropy = -1 * (pred * np.log(pred)).sum(axis=1)
        pred["entropy"] = entropy
        pred["prediction"] = pred.drop(columns="entropy").idxmax(axis=1)

        # add columns if available
        if self.col_datasets:
            pred[self.col_datasets] = self.df[self.col_datasets]
        if self.col_id_external:
            pred[self.col_id_external] = self.df[self.col_id_external]
        if self.col_label:
            pred["label"] = self.df[self.col_label]
        return pred

    def __compute_statistics(self, pred: DataFrame) -> dict[str, MLStatisticsModel]:
        """
        Compute statistics for the predictions
        """
        if self.df is None:
            raise ValueError("Dataframe is required to compute statistics")
        if self.statistics is None:
            raise ValueError("Statistics list is required to compute statistics")

        # compute statistics
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
                texts=pred[filter]["text"],
            )

        # add out of sample (labelled data not in training data)
        index_model = pd.read_parquet(self.path.joinpath("training_data.parquet"), columns=[]).index
        filter_oos = (
            ~pred.index.isin(index_model) & filter_label & pred[self.col_datasets] == "train"
        )
        if filter_oos.sum() > 10:
            metrics["outofsample"] = get_metrics(
                pred[filter_oos]["label"],
                pred[filter_oos]["prediction"],
                texts=pred[filter_oos]["text"],
            )

        # write the metrics in a json file
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)

        return metrics

    def __call__(self) -> ReturnTaskPredictModel:
        """
        Main process to predict
        """
        print("start predicting")

        if self.df is None:
            raise ValueError("Dataframe is required for prediction")

        # load the model
        tokenizer, model, max_length = self.__load_model()

        # check if GPU available
        gpu = False
        if torch.cuda.is_available():
            print("GPU is available")
            torch.cuda.empty_cache()
            gpu = True
            model.cuda()

        try:
            # prediction by batches
            predictions = []
            for i in range(0, self.df.shape[0], self.batch):
                print("Next chunck prediction")
                self.__listen_stop_event()
                chunk = tokenizer(
                    list(self.df[self.col_text][i : i + self.batch]),
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
                self.__write_progress((len(predictions) * self.batch / self.df.shape[0]) * 100)

            # transform predictions to clean dataframe
            pred = self.__transform_to_dataframe(
                predictions, columns=sorted(list(model.config.label2id.keys()))
            )

            # save the prediction to file
            pred.to_parquet(self.path.joinpath(self.file_name))

            # compute statistics if required
            if self.statistics:
                metrics = self.__compute_statistics(pred)
            else:
                metrics = None

        except Exception as e:
            print("Error in prediction", e)
            raise e
        finally:
            # delete the temporary logs
            os.remove(self.progress_path)
            # clean memory
            del tokenizer, model, chunk, self.df, res, predictions, outputs, self.event
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

        return ReturnTaskPredictModel(
            path=str(self.path.joinpath(self.file_name)), metrics=metrics
        )