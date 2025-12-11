import datetime
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]
from sklearn.model_selection import (  # type: ignore[import]
    KFold,
    cross_val_predict,
    train_test_split,
)

from activetigger.datamodels import QuickModelComputed
from activetigger.functions import get_metrics
from activetigger.tasks.base_task import BaseTask


class TrainML(BaseTask):
    """
    Fit a sklearn model
    """

    kind = "train_ml"

    def __init__(
        self,
        model: BaseEstimator,
        X: pd.DataFrame,
        Y: pd.Series,
        path: Path,
        name: str,
        user: str,
        model_params: dict,
        scheme: str,
        features: list,
        labels: list,
        model_type: str,
        standardize: bool = False,
        cv10: bool = False,
        balance_classes: bool = False,
        retrain: bool = False,
        texts: pd.Series | None = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.name = name
        self.X = X
        self.Y = Y
        self.user = user
        self.cv10 = cv10
        self.balance_classes = balance_classes
        self.path = path
        self.model_path = path.joinpath(name)
        self.retrain = retrain
        self.model_params = model_params
        self.scheme = scheme
        self.features = features
        self.labels = labels
        self.model_type = model_type
        self.standardize = standardize
        self.texts = texts

    def __call__(self) -> None:
        """
        Fit quickmodel and calculate statistics
        """

        # drop NA values
        f = self.Y.notnull()
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X[f], self.Y[f], test_size=0.2, random_state=42
        )

        # fit model
        self.model.fit(X_train, Y_train)

        # predict on test data
        Y_pred_train = pd.Series(self.model.predict(X_train), index=X_train.index)
        Y_pred_test = pd.Series(self.model.predict(X_test), index=X_test.index)

        # compute probabilities for all data
        proba = self.model.predict_proba(self.X)
        proba = pd.DataFrame(proba, columns=self.model.classes_, index=self.X.index)
        proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)
        proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)

        # compute training metrics and write
        if self.texts is not None:
            texts_train = self.texts.loc[Y_train.index]
            texts_test = self.texts.loc[Y_test.index]
            metrics_train = get_metrics(Y_train, Y_pred_train, texts=texts_train)
            metrics_test = get_metrics(Y_test, Y_pred_test, texts=texts_test)
        else:
            metrics_train = get_metrics(Y_train, Y_pred_train)
            metrics_test = get_metrics(Y_test, Y_pred_test)
            metrics_train.false_predictions = None
            metrics_test.false_predictions = None

        # compute 10-CV metrics
        if self.cv10:
            num_folds = 10
            kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
            Y_pred_10cv = cross_val_predict(self.model, self.X[f], self.Y[f], cv=kf)
            statistics_cv10 = get_metrics(self.Y[f], Y_pred_10cv)
            statistics_cv10.false_predictions = None
        else:
            statistics_cv10 = None

        # if retrain, clear the folder
        if self.retrain:
            shutil.rmtree(self.model_path)
            os.mkdir(self.model_path)
        else:
            if self.model_path.exists():
                raise Exception("The model already exists")
            os.mkdir(self.model_path)

        # Write the proba
        proba.to_csv(self.model_path / "proba.csv")

        # Write the training data
        X_train["label"] = Y_train
        X_train.to_parquet(self.model_path / "training_data.parquet")

        # Dump it in the folder
        element = QuickModelComputed(
            time=datetime.datetime.now(),
            model=self.model,
            user=self.user,
            name=self.name,
            scheme=self.scheme,
            features=self.features,
            labels=self.labels,
            model_type=self.model_type,
            model_params=self.model_params,
            standardize=self.standardize,
            cv10=self.cv10,
            balance_classes=self.balance_classes,
            retrain=self.retrain,
            proba=proba,
            statistics_train=metrics_train,
            statistics_test=metrics_test,
            statistics_cv10=statistics_cv10,
        )
        with open(self.model_path / "model.pkl", "wb") as file:
            pickle.dump(element, file)

        # Write the statistics
        with open(
            str(self.path.joinpath(self.name).joinpath("metrics_training.json")), "w"
        ) as file:
            json.dump(
                {
                    "train": metrics_train.model_dump(mode="json"),
                    "trainvalid": metrics_test.model_dump(mode="json"),
                    "cv10": statistics_cv10.model_dump(mode="json") if statistics_cv10 else None,
                },
                file,
            )

        return None
