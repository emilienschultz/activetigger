import logging
import os
import pickle
import shutil
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import]
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.naive_bayes import MultinomialNB  # type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]

from activetigger.datamodels import (
    KnnParams,
    LassoParams,
    LiblinearParams,
    ModelDescriptionModel,
    Multi_naivebayesParams,
    RandomforestParams,
    SimpleModelComputing,
    SimpleModelModel,
    SimpleModelsProjectStateModel,
)
from activetigger.db.languagemodels import LanguageModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.queue import Queue
from activetigger.tasks.fit_model import FitModel


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
    computing: list
    loaded: dict
    language_models_service: LanguageModelsService

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        db_manager: DatabaseManager,
    ) -> None:
        """
        Init Simplemodels class
        """
        self.path: Path = path.joinpath("simplemodels")
        if not self.path.exists():
            os.mkdir(self.path)
        self.project_slug = project_slug
        self.language_models_service = db_manager.language_models_service
        self.computing = computing
        self.queue = queue

        # Models and default parameters
        self.available_models = {
            "liblinear": LiblinearParams(cost=1),
            "knn": KnnParams(n_neighbors=3),
            "randomforest": RandomforestParams(n_estimators=500, max_features=None),
            "lasso": LassoParams(C=32),
            "multi_naivebayes": Multi_naivebayesParams(alpha=1, fit_prior=True, class_prior=None),
        }

        self.loaded = {}

    def add(self, element: SimpleModelComputing, results) -> None:
        """
        Add computed model
        - content in filesystem
        - add it to the database

        Manage the specific case of retrain
        """

        # Add element from the computation
        element.time = datetime.now()
        element.model = results.model
        element.proba = results.proba
        element.statistics = results.statistics
        element.statistics_cv10 = results.statistics_cv10

        # Create the filesystem
        model_path = self.path.joinpath(element.name)

        # if retrain, clear the folder
        if element.retrain:
            shutil.rmtree(model_path)
            os.mkdir(model_path)
        else:
            if model_path.exists():
                raise Exception("The model already exists")
            os.mkdir(model_path)

        # Write the proba
        if element.proba is not None:
            element.proba.to_csv(model_path / "proba.csv")

        # Dump it in the folder
        with open(model_path / "model.pkl", "wb") as file:
            pickle.dump(element, file)

        # Add the entry in the database
        self.language_models_service.add_model(
            kind="simplemodel",
            name=element.name,
            user=element.user,
            project=self.project_slug,
            scheme=element.scheme or "default",
            params=element.model_params or {},
            path=str(model_path),
            status="trained",
        )

    def compute_simplemodel(
        self,
        project_slug: str,
        user: str,
        scheme: str,
        features: list,
        name: str,
        model_type: str,
        df: DataFrame,
        col_labels: str,
        col_features: list,
        standardize: bool = True,
        model_params: dict | None = None,
        cv10: bool = False,
        retrain: bool = False,
    ) -> None:
        """
        Add a new simplemodel for a user and a scheme
        """
        logger_simplemodel = logging.getLogger("simplemodel")
        logger_simplemodel.info("Intiating the computation process for the simplemodel")
        X, Y, labels = self.transform_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[model_type].dict()

        # Select model
        if model_type == "knn":
            params_knn = KnnParams(**model_params)
            model = KNeighborsClassifier(n_neighbors=int(params_knn.n_neighbors), n_jobs=-1)
            model_params = params_knn.model_dump()

        if model_type == "lasso":
            params_lasso = LassoParams(**model_params)
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=params_lasso.C, n_jobs=-1
            )
            model_params = params_lasso.model_dump()

        if model_type == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            params_lib = LiblinearParams(**model_params)
            model = LogisticRegression(penalty="l2", solver="lbfgs", C=params_lib.cost, n_jobs=-1)
            model_params = params_lib.model_dump()

        if model_type == "randomforest":
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

        if model_type == "multi_naivebayes":
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

        req = SimpleModelComputing(
            user=user,
            unique_id=unique_id,
            time=datetime.now(),
            kind="simplemodel",
            scheme=scheme,
            model_type=model_type,
            name=name,
            features=features,
            labels=labels,
            model_params=model_params,
            standardize=standardize,
            model=model,
            cv10=cv10,
            retrain=retrain,
        )
        self.computing.append(req)

    def available(self) -> dict[str, list[ModelDescriptionModel]]:
        """
        Return available models per scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "simplemodel")
        r = {}
        for m in existing:
            if m.scheme not in r:
                r[m.scheme] = []
            r[m.scheme].append(m)
        return r

    def get(self, name: str) -> SimpleModelModel:
        """
        Load the content of a specific model
        (cache in memory)
        """
        if not self.exists(name):
            raise Exception("The model does not exist")
        if name in self.loaded:
            return self.loaded[name]
        else:
            path = self.path.joinpath(name)
            if not path.exists():
                raise Exception("The model path does not exist")
            with open(path / "model.pkl", "rb") as file:
                sm: SimpleModelModel = pickle.load(file)
            return sm

    def get_prediction(self, name: str) -> DataFrame:
        """
        Get a specific simplemodel
        """
        sm = self.get(name)
        if sm.proba is None:
            raise ValueError("No probability available for this model")
        return sm.proba

    def training(self) -> dict[str, list[str]]:
        """
        Currently under training
        """
        return {e.user: list(e.scheme) for e in self.computing if e.kind == "simplemodel"}

    def exists(self, name: str) -> bool:
        """
        Test if a simplemodel exists for a user/scheme
        """
        existing = self.language_models_service.available_models(self.project_slug, "simplemodel")
        return name in [m.name for m in existing]

    def transform_data(
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
            scaler = StandardScaler()
            df = data[~f_na][col_predictors]
            df_stand = scaler.fit_transform(df)
            df_pred = pd.DataFrame(df_stand, columns=df.columns, index=df.index)
        else:
            df_pred = data[~f_na][col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label], df_pred], axis=1)

        # data for training
        Y = df[col_label]
        X = df[col_predictors]
        labels = Y.unique()

        return X, Y, labels

    def export_prediction(self, name: str, format: str = "csv") -> tuple[BytesIO, dict[str, str]]:
        """
        Function to export the prediction of a simplemodel
        """
        # get data
        table = self.get_prediction(name)
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

    def delete(self, name: str) -> None:
        """
        Delete a specific simplemodel
        """
        if not self.exists(name):
            raise Exception("The model does not exist")

        # delete from the database
        self.language_models_service.delete_model(self.project_slug, name)

        # delete from the filesystem
        model_path = self.path.joinpath(name)
        if model_path.exists():
            shutil.rmtree(model_path)

        # delete from the loaded cache
        if name in self.loaded:
            del self.loaded[name]

    # def loads(self) -> bool:
    #     """
    #     Load
    #     """
    #     if not (self.path / self.save_file).exists():
    #         return False
    #     with open(self.path / self.save_file, "rb") as file:
    #         self.existing = pickle.load(file)
    #     return True

    # def add(self, element: SimpleModelComputing, results) -> None:
    #     """
    #     Add simplemodel after computation in the list of existing simplemodels
    #     And save the element
    #     """

    #     element.model = results.model
    #     element.proba = results.proba
    #     element.statistics = results.statistics
    #     element.statistics_cv10 = results.statistics_cv10
    #     if element.user not in self.existing:
    #         self.existing[element.user] = {}
    #     self.existing[element.user][element.scheme] = element
    #     self.dumps()

    # def exists(self, user: str, scheme: str) -> bool:
    #     """
    #     Test if a simplemodel exists for a user/scheme
    #     """
    #     if user in self.existing:
    #         if scheme in self.existing[user]:
    #             return True
    #     return False

    # def training(self) -> dict[str, list[str]]:
    #     """
    #     Currently under training
    #     """
    #     return {e.user: list(e.scheme) for e in self.computing if e.kind == "simplemodel"}

    # def available(self) -> dict[str, dict[str, SimpleModelOutModel]]:
    #     """
    #     Available simplemodels
    #     """
    #     r: dict[str, dict[str, SimpleModelOutModel]] = {}
    #     for u in self.existing:
    #         r[u] = {}
    #         for s in self.existing[u]:
    #             sm = self.existing[u][s]
    #             r[u][s] = SimpleModelOutModel(
    #                 scheme=s,
    #                 username=u,
    #                 model=sm.name,
    #                 params=sm.model_params,
    #                 features=sm.features,
    #                 statistics=sm.statistics,
    #                 statistics_cv10=sm.statistics_cv10,
    #             )
    #     return r

    # def get(self, scheme: str, username: str) -> SimpleModelOutModel | None:
    #     """
    #     Get a specific simplemodel
    #     """
    #     if username in self.existing:
    #         if scheme in self.existing[username]:
    #             sm = self.existing[username][scheme]
    #             return SimpleModelOutModel(
    #                 model=sm.name,
    #                 params=sm.model_params,
    #                 features=sm.features,
    #                 statistics=sm.statistics,
    #                 statistics_cv10=sm.statistics_cv10,
    #                 scheme=scheme,
    #                 username=username,
    #             )
    #     return None

    # def get_model(self, username: str, scheme: str) -> SimpleModelComputing:
    #     """
    #     Select a specific model in the repo
    #     """
    #     if username not in self.existing:
    #         raise Exception("The user does not exist")
    #     if scheme not in self.existing[username]:
    #         raise Exception("The scheme does not exist")
    #     return self.existing[username][scheme]
