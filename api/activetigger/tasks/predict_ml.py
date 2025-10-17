from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]

from activetigger.tasks.base_task import BaseTask


class PredictML(BaseTask):
    """
    Predict with a sklearn model
    """

    kind = "predict_model"

    def __init__(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        file_name: str,
        path: Path,
        labels: pd.Series | None = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.X = X
        self.labels = labels
        self.file_name = file_name
        self.path = path

    def __call__(self) -> np.ndarray:
        """
        Fit simplemodel and calculate statistics
        """
        Y = self.model.predict(self.X)
        print("Predictions done", Y)
        Y = pd.DataFrame(Y, index=self.X.index)
        Y.to_parquet(self.path.joinpath(self.file_name), index=True)

        # Compute statistics if labels are provided
