import datetime
import json
import math
import os
import pickle
import shutil
from pathlib import Path

import pandas as pd
from sklearn.base import BaseEstimator  # type: ignore[import]
from sklearn.model_selection import (  # type: ignore[import]
    KFold,
    cross_val_predict,
)

from activetigger.datamodels import MLStatisticsModel, QuickModelComputed
from activetigger.functions import evaluate_entropy, get_metrics
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
        labels: list[str],
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

    def __init_paths(self, retrain: bool) -> None:
        """
        Create a directory for the files to be saved
        """
        # if retrain, clear the folder
        if retrain:
            shutil.rmtree(self.model_path)
            os.mkdir(self.model_path)
        else:
            if self.model_path.exists():
                raise Exception("The model already exists")
            os.mkdir(self.model_path)

    def __split_set(
        self, test_size: float = 0.2
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Remove null elements and return train/test splits
        (equivalent of train_test_split from sklearn)
        """
        f = self.Y.notnull()  # filter drop NA values
        index = self.X[f].copy().index.to_series()
        index = index.sample(frac=1.0, random_state=42)
        n_total = len(index)
        n_test = math.ceil(n_total * test_size)
        n_train = n_total - n_test
        index_train = index.head(n_train)
        index_test = index.tail(n_test)
        X_train = self.X[f].loc[index_train.index, :]
        Y_train = self.Y[f].loc[index_train.index]
        X_test = self.X[f].loc[index_test.index, :]
        Y_test = self.Y[f].loc[index_test.index]
        return X_train, X_test, Y_train, Y_test

    def __compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> MLStatisticsModel:
        """
        Compute metrics
        """
        if self.texts is not None:
            texts = self.texts.loc[y_true.index]
        metrics = get_metrics(y_true, y_pred, texts=texts, labels=self.labels)
        return metrics

    def __compute_cv10(self) -> MLStatisticsModel:
        """
        Compute cv (predict and compute metrics)
        """
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        f = self.Y.notnull()
        Y_pred_10cv = pd.Series(
            cross_val_predict(self.model, self.X[f], self.Y[f], cv=kf), index=self.Y.index
        )

        statistics_cv10 = get_metrics(
            self.Y,
            Y_pred_10cv,
            labels=self.labels,
        )
        # overwrite false_predictions
        statistics_cv10.false_predictions = None

        return statistics_cv10

    def __create_saving_files(
        self,
        proba: pd.DataFrame,
        X_train: pd.DataFrame,
        Y_train: pd.Series,
        metrics_train: MLStatisticsModel,
        metrics_test: MLStatisticsModel,
        statistics_cv10: MLStatisticsModel | None,
    ) -> None:
        """Add an entry in the data base and save the following files:
        - proba.csv with the probabilities
        - data using during training (training_data.parquet)
        - a pickle version of the database entry (#NOTE: AM: Artefact ?)
        - metrics for the training (train, trainvalid and cv10)
        """
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
        path_to_metrics_json = str(self.path.joinpath(self.name).joinpath("metrics_training.json"))
        with open(path_to_metrics_json, "w") as file:
            json.dump(
                {
                    "train": metrics_train.model_dump(mode="json"),
                    "trainvalid": metrics_test.model_dump(mode="json"),
                    "cv10": statistics_cv10.model_dump(mode="json") if statistics_cv10 else None,
                },
                file,
            )

    def __call__(self) -> None:
        """
        Fit quickmodel and calculate statistics
        """
        start = datetime.datetime.now()
        self.__init_paths(self.retrain)

        X_train, X_test, Y_train, Y_test = self.__split_set()

        # Fit model --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
        self.model.fit(X_train, Y_train)

        # predict on test data --- --- --- --- --- --- --- --- --- --- --- --- -
        Y_pred_train = pd.Series(self.model.predict(X_train), index=X_train.index)
        Y_pred_test = pd.Series(self.model.predict(X_test), index=X_test.index)

        # compute probabilities for all data
        proba_values = self.model.predict_proba(self.X)
        proba = pd.DataFrame(proba_values, columns=self.model.classes_, index=self.X.index)
        proba["prediction"] = proba.idxmax(axis=1)
        proba["entropy"] = evaluate_entropy(proba_values)

        # Compute metrics --- --- --- --- --- --- --- --- --- --- --- --- --- --
        metrics_train = self.__compute_metrics(y_true=Y_train, y_pred=Y_pred_train)
        metrics_test = self.__compute_metrics(y_true=Y_test, y_pred=Y_pred_test)

        if self.cv10:
            statistics_cv10 = self.__compute_cv10()
        else:
            statistics_cv10 = None

        self.__create_saving_files(
            proba, X_train, Y_train, metrics_train, metrics_test, statistics_cv10
        )

        end = datetime.datetime.now()
        print(f"Training completed in {(end - start).total_seconds()} seconds")
