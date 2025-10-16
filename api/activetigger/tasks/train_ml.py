import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]
from sklearn.model_selection import (  # type: ignore[import]
    KFold,
    cross_val_predict,
    train_test_split,
)

from activetigger.datamodels import TrainMLResults
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class TrainML(BaseTask):
    """
    Fit a sklearn model
    """

    kind = "train_ml"

    def __init__(
        self, model: BaseEstimator, X: pd.DataFrame, Y: pd.Series, cv10: bool = False, **kwargs
    ):
        super().__init__()
        self.model = model
        self.X = X
        self.Y = Y
        self.cv10 = cv10

    def __call__(self) -> TrainMLResults:
        """
        Fit simplemodel and calculate statistics
        """

        # drop NA values
        f = self.Y.notnull()
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X[f], self.Y[f], test_size=0.2, random_state=42
        )

        # fit model
        self.model.fit(X_train, Y_train)

        # compute statistics
        Y_pred = self.model.predict(X_test)
        statistics = get_metrics(Y_test, Y_pred)
        statistics.false_predictions = None

        # compute probabilities for all data
        proba = self.model.predict_proba(self.X)
        proba = pd.DataFrame(proba, columns=self.model.classes_, index=self.X.index)
        proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)
        proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)

        # compute 10-crossvalidation
        if self.cv10:
            num_folds = 10
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            Y_pred_10cv = cross_val_predict(self.model, self.X[f], self.Y[f], cv=kf)
            statistics_cv10 = get_metrics(self.Y[f], Y_pred_10cv)
            statistics_cv10.false_predictions = None
        else:
            statistics_cv10 = None

        return TrainMLResults(
            model=self.model,
            proba=proba,
            statistics=statistics,
            statistics_cv10=statistics_cv10,
        )
