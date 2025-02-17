import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import KFold, cross_val_predict

from activetigger.tasks.base_task import BaseTask


class FitModel(BaseTask):
    """
    Fit a sklearn model
    """

    kind = "fit_model"

    def __init__(self, model: BaseEstimator, X: pd.DataFrame, Y: pd.Series, **kwargs):
        self.model = model
        self.X = X
        self.Y = Y

    def __call__(self):
        """
        Fit simplemodel and calculate statistics
        """
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
        f1 = f1_score(Yf.values, Y_pred, average=None)
        weighted_f1 = f1_score(Yf, Y_pred, average="weighted")
        accuracy = accuracy_score(Yf, Y_pred)
        precision = precision_score(
            list(Yf),
            list(Y_pred),
            average="micro",
        )
        macro_f1 = f1_score(Yf, Y_pred, average="macro")
        statistics = {
            "f1": [round(i, 3) for i in list(f1)],
            "weighted_f1": round(weighted_f1, 3),
            "macro_f1": round(macro_f1, 3),
            "accuracy": round(accuracy, 3),
            "precision": round(precision, 3),
        }

        # compute 10-crossvalidation
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        Y_pred = cross_val_predict(self.model, Xf, Yf, cv=kf)
        weighted_f1 = f1_score(Yf, Y_pred, average="weighted")
        accuracy = accuracy_score(Yf, Y_pred)
        macro_f1 = f1_score(Yf, Y_pred, average="macro")
        cv10 = {
            "weighted_f1": round(weighted_f1, 3),
            "macro_f1": round(macro_f1, 3),
            "accuracy": round(accuracy, 3),
        }

        r = {
            "model": self.model,
            "proba": proba,
            "statistics": statistics,
            "cv10": cv10,
        }

        print("STATISTICS", statistics)

        return r
