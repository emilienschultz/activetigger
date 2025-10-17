from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.base import BaseEstimator

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
        col_labels: str | None = None,
        unique_id: Optional[str] = None,
        statistics: list | None = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.df = df
        self.file_name = file_name
        self.path = path
        self.unique_id = unique_id
        self.statistics = statistics

        if col_dataset not in self.df.columns:
            raise ValueError(f"Dataset column {col_dataset} not in dataframe")

        if not all(col in self.df.columns for col in col_features):
            raise ValueError("Some feature columns are not in dataframe")

        if self.statistics is not None and self.col_labels is None:
            raise ValueError("Labels must be provided to compute statistics")

        expected_cols = self.model.feature_names_in_
        missing = set(expected_cols) - set(col_features)
        extra = set(col_features) - set(expected_cols)
        if len(missing) > 0 or len(extra) > 0:
            raise ValueError(f"Feature mismatch. Missing: {missing}, Extra: {extra}")
        self.X = self.df.reindex(columns=list(expected_cols) + [col_dataset])
        self.Y = self.df[col_labels] if col_labels is not None else None

    def __call__(self):
        """
        Fit simplemodel and calculate statistics
        """
        # Predict
        Y_pred = self.model.predict(self.X.drop(columns=["dataset"]))
        Y_proba = self.model.predict_proba(self.X.drop(columns=["dataset"]))
        proba_cols = [cls for cls in self.model.classes_]
        Y_full = pd.DataFrame(Y_proba, columns=proba_cols, index=self.X.index)
        Y_full["pred_label"] = Y_pred
        Y_full["dataset"] = self.X["dataset"]
        Y_full.to_parquet(self.path.joinpath(self.file_name), index=True)
        print(Y_full.head())

        # Compute statistics if labels are provided
        print("Compute statistics")
        metrics = {}

        # # compute the statistics per dataset
        # filter_label = pred["label"].notna()  # only non null values
        # for dataset in self.statistics:
        #     filter_dataset = Y_full["dataset"] == dataset
        #     filter = filter_label & filter_dataset
        #     if filter.sum() < 5:
        #         continue
        #     metrics[dataset] = get_metrics(
        #         pred[filter]["label"],
        #         pred[filter]["prediction"],
        #         pred["text"],
        #     )

        # # write the metrics in a json file
        # with open(str(self.path.joinpath(f"metrics_predict_{time.time()}.json")), "w") as f:
        #     json.dump({k: v.model_dump(mode="json") for k, v in metrics.items()}, f)
