import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]
from sklearn.model_selection import KFold, cross_val_predict  # type: ignore[import]

from activetigger.datamodels import FitModelResults
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class FitModel(BaseTask):
    """
    Fit a sklearn model
    """

    kind = "fit_model"

    def __init__(
        self, model: BaseEstimator, X: pd.DataFrame, Y: pd.Series, cv10: bool = False, **kwargs
    ):
        super().__init__()
        self.model = model
        self.X = X
        self.Y = Y
        self.cv10 = cv10

    def __call__(self) -> FitModelResults:
        """
        Fit simplemodel and calculate statistics
        """

        print("start fit model")

        # drop NA values
        f = self.Y.notnull()
        Xf = self.X[f]
        Yf = self.Y[f]

        # fit model
        self.model.fit(Xf, Yf)

        # compute probabilities
        proba = self.model.predict_proba(self.X)
        proba = pd.DataFrame(proba, columns=self.model.classes_, index=self.X.index)
        proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)
        proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)

        # compute statistics
        Y_pred = self.model.predict(Xf)

        statistics = get_metrics(Yf, Y_pred)

        # compute 10-crossvalidation
        if self.cv10:
            num_folds = 10
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            Y_pred_10cv = cross_val_predict(self.model, Xf, Yf, cv=kf)
            cv10 = get_metrics(Yf, Y_pred_10cv)
        else:
            cv10 = None

        return FitModelResults(
            model=self.model,
            proba=proba,
            statistics=statistics,
            cv10=cv10,
        )
