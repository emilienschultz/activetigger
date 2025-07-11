import logging
import pickle
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
from pandas import DataFrame
from pydantic import BaseModel, ConfigDict
from sklearn.base import BaseEstimator  # type: ignore[import]
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.naive_bayes import MultinomialNB  # type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]

from activetigger.config import config
from activetigger.datamodels import (
    KnnParams,
    LassoParams,
    LiblinearParams,
    MLStatisticsModel,
    Multi_naivebayesParams,
    RandomforestParams,
    SimpleModelComputing,
    SimpleModelOutModel,
    SimpleModelsProjectStateModel,
)
from activetigger.queue import Queue
from activetigger.tasks.fit_model import FitModel


class SimpleModel(BaseModel):
    """
    Simplemodel object
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    user: str
    features: list
    labels: list
    model_params: dict
    standardize: bool
    model: BaseEstimator
    proba: DataFrame | None = None
    statistics: MLStatisticsModel | None = None
    statistics_cv10: MLStatisticsModel | None = None


class SimpleModels:
    """
    Module to manage simplemodels
    - define available models
    - save a simplemodel/user
    - train simplemodels
    """

    path: Path
    queue: Queue
    available_models: dict[str, Any]
    existing: dict[str, dict[str, SimpleModelComputing]]
    computing: list
    save_file: str

    def __init__(self, path: Path, queue: Queue, computing: list) -> None:
        """
        Init Simplemodels class
        """
        # Models and default parameters
        self.available_models = {
            "liblinear": LiblinearParams(cost=1),
            "knn": KnnParams(n_neighbors=3),
            "randomforest": RandomforestParams(n_estimators=500, max_features=None),
            "lasso": LassoParams(C=32),
            "multi_naivebayes": Multi_naivebayesParams(alpha=1, fit_prior=True, class_prior=None),
        }

        self.existing: dict = {}  # computed simplemodels for users / schemes DICT > USERS > SCHEME
        self.computing: list = computing  # currently under computation
        self.path: Path = path  # path to operate
        self.queue = queue  # access to executor for multiprocessing
        self.save_file: str = config.simplemodels_file  # file to save current state
        self.loads()  # load existing simplemodels for the project

    def __repr__(self) -> str:
        return str(self.available())

    def available(self) -> dict[str, dict[str, SimpleModelOutModel]]:
        """
        Available simplemodels
        """
        r: dict[str, dict[str, SimpleModelOutModel]] = {}
        for u in self.existing:
            r[u] = {}
            for s in self.existing[u]:
                sm = self.existing[u][s]
                r[u][s] = SimpleModelOutModel(
                    scheme=s,
                    username=u,
                    model=sm.name,
                    params=sm.model_params,
                    features=sm.features,
                    statistics=sm.statistics,
                    statistics_cv10=sm.statistics_cv10,
                )
        return r

    def get(self, scheme: str, username: str) -> SimpleModelOutModel | None:
        """
        Get a specific simplemodel
        """
        if username in self.existing:
            if scheme in self.existing[username]:
                sm = self.existing[username][scheme]
                return SimpleModelOutModel(
                    model=sm.name,
                    params=sm.model_params,
                    features=sm.features,
                    statistics=sm.statistics,
                    statistics_cv10=sm.statistics_cv10,
                    scheme=scheme,
                    username=username,
                )
        return None

    def get_prediction(self, scheme: str, username: str) -> DataFrame:
        """
        Get a specific simplemodel
        """
        if username not in self.existing:
            raise ValueError("No model for this user")
        if scheme not in self.existing[username]:
            raise ValueError("No model for this scheme")
        if self.existing[username][scheme].proba is None:
            raise ValueError("No prediction available for this model")
        return self.existing[username][scheme].proba

    def training(self) -> dict[str, list[str]]:
        """
        Currently under training
        """
        return {e.user: list(e.scheme) for e in self.computing if e.kind == "simplemodel"}

    def exists(self, user: str, scheme: str) -> bool:
        """
        Test if a simplemodel exists for a user/scheme
        """
        if user in self.existing:
            if scheme in self.existing[user]:
                return True
        return False

    def get_model(self, user: str, scheme: str) -> SimpleModelComputing:
        """
        Select a specific model in the repo
        """
        if user not in self.existing:
            raise Exception("The user does not exist")
        if scheme not in self.existing[user]:
            raise Exception("The scheme does not exist")
        return self.existing[user][scheme]

    def load_data(
        self, data, col_label, col_predictors, standardize
    ) -> tuple[DataFrame, DataFrame, list]:
        """
        Load data
        """
        f_na = data[col_predictors].isna().sum(axis=1) > 0
        if f_na.sum() > 0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize X data
        if standardize:
            df_pred = self.standardize(data[~f_na][col_predictors])
        else:
            df_pred = data[~f_na][col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label], df_pred], axis=1)

        # data for training
        Y = df[col_label]
        X = df[col_predictors]
        labels = Y.unique()

        return X, Y, labels

    def standardize(self, df) -> DataFrame:
        """
        Apply standardization
        """
        scaler = StandardScaler()
        df_stand = scaler.fit_transform(df)
        return pd.DataFrame(df_stand, columns=df.columns, index=df.index)

    def compute_simplemodel(
        self,
        project_slug: str,
        user: str,
        scheme: str,
        features: list,
        name: str,
        df: DataFrame,
        col_labels: str,
        col_features: list,
        standardize: bool = True,
        model_params: dict | None = None,
        cv10: bool = False,
    ) -> None:
        """
        Add a new simplemodel for a user and a scheme
        """
        logger_simplemodel = logging.getLogger("simplemodel")
        logger_simplemodel.info("Intiating the computation process for the simplemodel")
        X, Y, labels = self.load_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[name].dict()

        # Select model
        if name == "knn":
            params_knn = KnnParams(**model_params)
            model = KNeighborsClassifier(n_neighbors=int(params_knn.n_neighbors), n_jobs=-1)
            model_params = params_knn.model_dump()

        if name == "lasso":
            params_lasso = LassoParams(**model_params)
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=params_lasso.C, n_jobs=-1
            )
            model_params = params_lasso.model_dump()

        if name == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            params_lib = LiblinearParams(**model_params)
            model = LogisticRegression(penalty="l2", solver="lbfgs", C=params_lib.cost, n_jobs=-1)
            model_params = params_lib.model_dump()

        if name == "randomforest":
            # params  Num. trees mtry  Sample fraction
            # Number of variables randomly sampled as candidates at each split:
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            params_rf = RandomforestParams(**model_params)
            model = RandomForestClassifier(
                n_estimators=int(params_rf.n_estimators),
                random_state=42,
                max_features=(
                    int(params_rf.max_features) if params_rf.max_features is not None else None
                ),
                n_jobs=-1,
            )
            model_params = params_rf.model_dump()

        if name == "multi_naivebayes":
            # small workaround for parameters
            params_nb = Multi_naivebayesParams(**model_params)
            if params_nb.class_prior is not None:
                class_prior = params_nb.class_prior
            else:
                class_prior = None
            # Only with dtf or tfidf for features
            # TODO: calculate class prior for docfreq & termfreq
            model = MultinomialNB(
                alpha=params_nb.alpha,
                fit_prior=params_nb.fit_prior,
                class_prior=class_prior,
            )
            model_params = params_nb.model_dump()

        # launch the compuation (model + statistics) as a future process
        args = {
            "model": model,
            "X": X,
            "Y": Y,
            "labels": labels,
            "cv10": cv10,
        }
        unique_id = self.queue.add_task("simplemodel", project_slug, FitModel(**args))
        del args

        # add features in the params
        model_params["features"] = features

        self.computing.append(
            SimpleModelComputing(
                user=user,
                unique_id=unique_id,
                time=datetime.now(),
                kind="simplemodel",
                scheme=scheme,
                name=name,
                features=features,
                labels=labels,
                model_params=model_params,
                standardize=standardize,
                model=model,
                cv10=cv10,
            )
        )

    def dumps(self) -> None:
        """
        Dumps all simplemodels to a pickle
        """
        with open(self.path / self.save_file, "wb") as file:
            pickle.dump(self.existing, file)

    def loads(self) -> bool:
        """
        Load all simplemodels from a pickle
        """
        if not (self.path / self.save_file).exists():
            return False
        with open(self.path / self.save_file, "rb") as file:
            self.existing = pickle.load(file)
        return True

    def add(self, element: SimpleModelComputing, results) -> None:
        """
        Add simplemodel after computation in the list of existing simplemodels
        And save the element
        """

        element.model = results.model
        element.proba = results.proba
        element.statistics = results.statistics
        element.statistics_cv10 = results.statistics_cv10
        if element.user not in self.existing:
            self.existing[element.user] = {}
        self.existing[element.user][element.scheme] = element
        self.dumps()

    def export_prediction(
        self, scheme: str, username: str, format: str = "csv"
    ) -> tuple[BytesIO, dict[str, str]]:
        """
        Function to export the prediction of a simplemodel
        """
        # get data
        table = self.get_prediction(scheme, username)
        # convert to payload
        if format == "csv":
            output = BytesIO()
            pd.DataFrame(table).to_csv(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.csv"',
                "Content-Type": "text/csv",
            }
            return output, headers
        elif format == "xlsx":
            output = BytesIO()
            pd.DataFrame(table).to_excel(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.xlsx"',
                "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            }
            return output, headers
        elif format == "parquet":
            output = BytesIO()
            pd.DataFrame(table).to_parquet(output)
            output.seek(0)
            headers = {
                "Content-Disposition": 'attachment; filename="data.parquet"',
                "Content-Type": "application/octet-stream",
            }
            return output, headers
        else:
            raise ValueError("Format not supported")

    def state(self) -> SimpleModelsProjectStateModel:
        return SimpleModelsProjectStateModel(
            options=self.available_models,
            available=self.available(),
            training=self.training(),
        )
