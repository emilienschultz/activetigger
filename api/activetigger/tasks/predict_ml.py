import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]

from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class PredictML(BaseTask):
    """
    Predict with a sklearn model
    """

    kind = "predict_model"

    def __init__(
        self,
        model: BaseEstimator,
        df: pd.DataFrame,
        col_dataset: str,
        col_features: list[str],
        file_name: str,
        path: Path,
        col_label: str | None = None,
        col_text: str | None = None,
        unique_id: Optional[str] = None,
        statistics: list[str] | None = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.df = df
        self.file_name = file_name
        self.path = path
        self.unique_id = unique_id
        self.statistics = statistics
        self.col_label = col_label
        self.col_text = col_text

        if col_dataset not in self.df.columns:
            raise ValueError(f"Dataset column {col_dataset} not in dataframe")

        if not all(col in self.df.columns for col in col_features):
            raise ValueError("Some feature columns are not in dataframe")

        if self.statistics is not None and self.col_label is None:
            raise ValueError("Labels must be provided to compute statistics")

        expected_cols = self.model.feature_names_in_
        missing = set(expected_cols) - set(col_features)
        extra = set(col_features) - set(expected_cols)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")
        self.X = self.df.reindex(columns=list(expected_cols) + [col_dataset])
        self.Y = self.df[col_label] if col_label is not None else None

        if self.statistics and self.col_text is None:
            print("There is no full text")

    def __call__(self):
        """
        Fit simplemodel and calculate statistics
        """
        # Predict
        Y_pred = self.model.predict(self.X.drop(columns=["dataset"]))
        Y_proba = self.model.predict_proba(self.X.drop(columns=["dataset"]))
        proba_cols = [cls for cls in self.model.classes_]
        Y_full = pd.DataFrame(Y_proba, columns=proba_cols, index=self.X.index)
        Y_full["prediction"] = Y_pred
        Y_full["dataset"] = self.X["dataset"]
        Y_full["label"] = self.df[self.col_label]
        Y_full.to_parquet(self.path.joinpath(self.file_name), index=True)
        if self.col_text:
            Y_full["text"] = self.df[self.col_text]

        # Compute statistics if labels are provided
        metrics = {}

        filter_label = self.df[self.col_label].notna()  # only non null values
        for dataset in self.statistics:
            filter_dataset = Y_full["dataset"] == dataset
            filter = filter_label & filter_dataset
            if filter.sum() < 5:
                continue
            metrics[dataset] = get_metrics(
                Y_full[filter]["label"],
                Y_full[filter]["prediction"],
                Y_full[filter]["text"] if self.col_text else None,
            )

        # add out of sample (labelled data not in training data)
        index_model = pd.read_parquet(self.path.joinpath("training_data.parquet"), columns=[]).index
        filter_oos = ~Y_full.index.isin(index_model) & filter_label & Y_full["dataset"] == "train"
        if filter_oos.sum() > 10:
            metrics["outofsample"] = get_metrics(
                Y_full[filter_oos]["label"],
                Y_full[filter_oos]["prediction"],
                Y_full[filter_oos]["text"] if self.col_text else None,
            )

        # write the metrics in a json file
        with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
            json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)
