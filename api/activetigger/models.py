import json
import logging
import os
import pickle
import shutil
from datetime import datetime
from io import BytesIO
from multiprocessing import Process
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import]
from sklearn.linear_model import LogisticRegression  # type: ignore[import]
from sklearn.naive_bayes import MultinomialNB  # type: ignore[import]
from sklearn.neighbors import KNeighborsClassifier  # type: ignore[import]
from sklearn.preprocessing import StandardScaler  # type: ignore[import]
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import]

import activetigger.functions as functions
from activetigger.datamodels import (
    BertModelInformationsModel,
    BertModelParametersDbModel,
    BertModelParametersModel,
    KnnParams,
    LassoParams,
    LiblinearParams,
    MLStatisticsModel,
    Multi_naivebayesParams,
    RandomforestParams,
    SimpleModelOutModel,
    StaticFileModel,
    UserModelComputing,
)
from activetigger.db.manager import DatabaseManager
from activetigger.db.projects import ProjectsService
from activetigger.queue import Queue
from activetigger.tasks.fit_model import FitModel
from activetigger.tasks.predict_bert import PredictBert
from activetigger.tasks.train_bert import TrainBert


class BertModel:
    """
    Manage one bertmodel
    """

    name: str
    path: Path
    params: dict | None
    base_model: str | None
    tokenizer = None
    model = None
    log_history = None
    status: str
    pred: DataFrame | None
    data: DataFrame | None
    timestamp: datetime

    def __init__(
        self,
        name: str,
        path: Path,
        base_model: str | None = None,
        params: dict | None = None,
    ) -> None:
        """
        Init a bert model
        """
        self.name = name
        self.path = path
        self.params = params
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.log_history = None
        self.status = "initializing"
        self.pred = None
        self.data = None
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"{self.name} - {self.base_model}"

    def load(self, lazy=False):
        """
        Load trained model from files
        - either lazy (only parameters)
        - or complete (the weights of the model)
        """
        if not (self.path.joinpath("config.json")).exists():
            raise FileNotFoundError("model not defined")

        # Load parameters
        with open(self.path.joinpath("parameters.json"), "r") as jsonfile:
            self.params = json.load(jsonfile)

        # Load training data
        self.data = pd.read_parquet(self.path.joinpath("training_data.parquet"))

        # Load train history
        with open(self.path.joinpath("log_history.txt"), "r") as f:
            self.log_history = json.load(f)

        # Load prediction if available
        if (self.path.joinpath("predict_train.parquet")).exists():
            self.pred = pd.read_parquet(self.path.joinpath("predict_train.parquet"))

        with open(self.path.joinpath("config.json"), "r") as jsonfile:
            modeltype = json.load(jsonfile)["_name_or_path"]
        self.base_model = modeltype

        # Only load the model if not lazy mode
        if lazy:
            self.status = "lazy"
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(modeltype)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
            self.status = "loaded"

    def get_labels(self):
        with open(self.path.joinpath("config.json"), "r") as f:
            r = json.load(f)
        return list(r["id2label"].values())

    def get_training_progress(self) -> float | None:
        """
        Get progress when training
        (different cases)
        """
        # case of training
        if (self.status == "training") & (
            self.path.joinpath("train/progress")
        ).exists():
            with open(self.path.joinpath("train/progress"), "r") as f:
                r = f.read()
                if r == "":
                    r = "0"
            return float(r)
        # case for prediction (predicting/testing)
        if (("predicting" in self.status) or (self.status == "testing")) & (
            self.path.joinpath("progress_predict")
        ).exists():
            with open(self.path.joinpath("progress_predict"), "r") as f:
                r = f.read()
                if r == "":
                    r = "0"
            return float(r)
        return None

    def get_loss(self) -> dict | None:
        try:
            with open(self.path.joinpath("log_history.txt"), "r") as f:
                log = json.load(f)
            loss = pd.DataFrame(
                [
                    [
                        log[2 * i]["epoch"],
                        log[2 * i]["loss"],
                        log[2 * i + 1]["eval_loss"],
                    ]
                    for i in range(0, int((len(log) - 1) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            ).to_json()
            return json.loads(loss)
        except Exception:
            return None

    def informations(self) -> BertModelInformationsModel:
        """
        Informations on the bert model from the files
        TODO : avoid to read and create a cache
        """

        loss = self.get_loss()

        # train scores
        if (self.path.joinpath("metrics_predict_train.parquet.json")).exists():
            with open(
                self.path.joinpath("metrics_predict_train.parquet.json"), "r"
            ) as f:
                train_scores = json.load(f)
        else:
            train_scores = None

        # test scores
        if (self.path.joinpath("metrics_predict_test.parquet.json")).exists():
            with open(
                self.path.joinpath("metrics_predict_test.parquet.json"), "r"
            ) as f:
                test_scores = json.load(f)
        else:
            test_scores = None

        return BertModelInformationsModel(
            params=self.params,
            loss=loss,
            train_scores=train_scores,
            test_scores=test_scores,
        )


class BertModels:
    """
    Managing bertmodel training
    """

    project_slug: str
    path: Path
    queue: Any
    computing: list[UserModelComputing]
    projects_service: ProjectsService
    db_manager: DatabaseManager
    base_models: list
    params_default: BertModelParametersModel

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Any,
        computing: list,
        db_manager: DatabaseManager,
        list_models: str | None = None,
    ) -> None:
        self.params_default = BertModelParametersModel()

        # load the list of models
        if list_models is not None:
            self.base_models = pd.read_csv(list_models).to_dict(orient="records")
        else:
            self.base_models = [
                {
                    "name": "answerdotai/ModernBERT-base",
                    "priority": 10,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "camembert/camembert-base",
                    "priority": 0,
                    "comment": "",
                    "language": "fr",
                },
                {
                    "name": "camembert/camembert-large",
                    "priority": 0,
                    "comment": "",
                    "language": "fr",
                },
                {
                    "name": "flaubert/flaubert_small_cased",
                    "priority": 5,
                    "comment": "",
                    "language": "fr",
                },
                {
                    "name": "flaubert/flaubert_base_cased",
                    "priority": 7,
                    "comment": "",
                    "language": "fr",
                },
                {
                    "name": "flaubert/flaubert_large_cased",
                    "priority": 9,
                    "comment": "",
                    "language": "fr",
                },
                {
                    "name": "distilbert-base-cased",
                    "priority": 0,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "roberta-base",
                    "priority": 0,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "microsoft/deberta-base",
                    "priority": 0,
                    "comment": "",
                    "language": "en",
                },
                {
                    "name": "distilbert-base-multilingual-cased",
                    "priority": 0,
                    "comment": "",
                    "language": "multi",
                },
                {
                    "name": "microsoft/Multilingual-MiniLM-L12-H384",
                    "priority": 0,
                    "comment": "",
                    "language": "multi",
                },
            ]
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.projects_service = db_manager.projects_service
        self.path: Path = Path(path) / "bert"
        if not self.path.exists():
            os.mkdir(self.path)

    def __repr__(self) -> str:
        return f"Trained models : {self.available()}"

    def available(self) -> dict:
        """
        Information on available models for state
        """
        models = self.projects_service.available_models(self.project_slug)
        r: dict = {}
        for m in models:
            if m["scheme"] not in r:
                r[m["scheme"]] = {}
            r[m["scheme"]][m["name"]] = {
                "predicted": m["parameters"]["predicted"],
                "compressed": m["parameters"]["compressed"],
            }
            # if no compression, start it
            if not m["parameters"]["compressed"]:
                if (self.path / "../../static" / f"{m['name']}.tar.gz").exists():
                    # update bdd
                    self.projects_service.set_model_params(
                        self.project_slug,
                        m["name"],
                        "compressed",
                        True,
                    )
                else:
                    # create a flag
                    with open(
                        self.path / "../../static" / f"{m['name']}.tar.gz", "w"
                    ) as f:
                        f.write("process started")
                    # start compression
                    self.start_compression(m["name"])

        return r

    def training(self) -> dict:
        """
        Currently under training
        - name
        - progress if available
        - loss if available
        """
        r = {
            e.user: {
                "name": e.model_name,
                "status": e.status,
                "progress": (
                    e.get_training_progress() if e.get_training_progress else None
                ),
                "loss": e.model.get_loss(),
            }
            for e in self.computing
            if e.kind in ["bert", "train_bert", "predict_bert"]
        }
        return r

    def delete(self, bert_name: str) -> dict:
        """
        Delete bert model
        """
        r = self.projects_service.delete_model(self.project_slug, bert_name)
        if not r:
            raise FileNotFoundError("Model does not exist")
        try:
            shutil.rmtree(self.path / bert_name)
            os.remove(self.path / "../../static" / f"{bert_name}.tar.gz")
        except Exception as e:
            raise Exception(f"Problem to delete model : {e}")
        return {"success": "Bert model deleted"}

    def current_user_processes(self, user: str) -> list[UserModelComputing]:
        """
        Get the user current processes
        """
        return [e for e in self.computing if e.user == user]

    def estimate_memory_use(self, model: str, kind: str = "train"):
        """
        Estimate the GPU memory in Gb needed to train a model
        TODO : implement
        """
        if kind == "train":
            return 4
        if kind == "predict":
            return 3

    def start_training_process(
        self,
        name: str,
        project: str,
        user: str,
        scheme: str,
        df: DataFrame,
        col_text: str,
        col_label: str,
        params: BertModelParametersModel,
        base_model: str = "almanach/camembert-base",
        test_size: float = 0.2,
    ) -> dict:
        """
        Manage the training of a model from the API
        """
        # Check if there is no other competing processes : 1 active process by user
        if len(self.current_user_processes(user)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

        # check the size of training data
        if len(df.dropna()) < 10:
            raise Exception("Less than 10 elements annotated")

        # check the number of elements
        counts = df[col_label].value_counts()
        if not (counts >= 5).all():
            raise Exception("Less than 5 elements per label")

        # name integrating the scheme & user + date
        current_date = datetime.now()
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")
        name = f"{name}__{user}__{project}__{scheme}__{day}-{month}-{year}"

        # check if a project not already exist
        if self.projects_service.model_exists(project, name):
            raise Exception("A model with this name already exists")

        # if GPU requested, test if enough memory is available (to avoid CUDA out of memory)
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if self.estimate_memory_use(name, kind="train") > mem["available_memory"]:
                raise Exception(
                    "Not enough GPU memory available. Wait or reduce batch."
                )

        # launch as a independant process
        unique_id = self.queue.add_task(
            "training",
            project,
            TrainBert(
                path=self.path,
                name=name,
                df=df.copy(deep=True),
                col_label=col_label,
                col_text=col_text,
                base_model=base_model,
                params=params,
                test_size=test_size,
            ),
        )
        # unique_id  = self.queue.add(
        #     "training",
        #     project,
        #     train_bert,
        #     {
        #         "path": self.path,
        #         "name": name,
        #         "df": df.copy(deep=True),
        #         "col_label": col_label,
        #         "col_text": col_text,
        #         "base_model": base_model,
        #         "params": params,
        #         "test_size": test_size,
        #     },
        # )

        # Update the queue state
        b = BertModel(name, self.path / name, base_model)
        b.status = "training"
        self.computing.append(
            UserModelComputing(
                user=user,
                model=b,
                model_name=b.name,
                unique_id=unique_id,
                time=current_date,
                kind="train_bert",
                status="training",
                scheme=scheme,
                dataset=None,
                get_training_progress=b.get_training_progress,
            )
        )

        # add flags in params
        params = BertModelParametersDbModel(**params.model_dump())

        # add in database
        if not self.projects_service.add_model(
            kind="bert",
            name=name,
            user=user,
            project=project,
            scheme=scheme,
            params=params.model_dump(),
            path=str(self.path / name),
            status="training",
        ):
            raise Exception("Problem to add in database")

        return {"success": "bert model on training"}

    def start_testing_process(
        self,
        project_slug: str,
        name: str,
        user: str,
        df: DataFrame,
        col_text: str,
        col_labels: str,
    ):
        """
        Start testing process
        - launch as an independant process functions.test_bert
        - once computed, sync with the queue
        """
        if len(self.current_user_processes(user)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

        if not (self.path / name).exists():
            raise Exception("The model does not exist")

        # test number of elements in the test set
        if len(df["labels"].dropna()) < 10:
            raise Exception("Less than 10 elements annotated")

        # load model
        b = BertModel(name, self.path / name)
        b.load(lazy=True)

        # test if the testset and the model have the same labels
        labels_model = set(b.get_labels())
        labels_test = set(df["labels"].dropna().unique())
        if set(b.get_labels()) != set(df["labels"].dropna().unique()):
            raise Exception(
                f"The testset and the model have different labels {labels_model} vs {labels_test}"
            )

        # delete previous files
        if (self.path.joinpath(name).joinpath("predict_test.parquet")).exists():
            os.remove(self.path.joinpath(name).joinpath("predict_test.parquet"))
        if (self.path.joinpath(name).joinpath("statistics.json")).exists():
            os.remove(self.path.joinpath(name).joinpath("statistics.json"))

        # start prediction on the test set

        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictBert(
                df=df,
                col_text=col_text,
                col_label=col_labels,
                path=b.path,
                basemodel=b.base_model,
                file_name="predict_test.parquet",
                batch=32,
            ),
        )

        b.status = "testing"
        self.computing.append(
            UserModelComputing(
                user=user,
                model=b,
                model_name=b.name,
                unique_id=unique_id,
                time=datetime.now(),
                kind="predict_bert",
                status="testing",
                get_training_progress=b.get_training_progress,
                dataset="test",
            )
        )

        return {"success": "bert testing predicting"}

    def start_predicting_process(
        self,
        project_slug: str,
        name: str,
        user: str,
        df: DataFrame,
        col_text: str,
        dataset: str,
        col_label: str | None = None,
        batch_size: int = 32,
    ):
        """
        Start predicting process
        """
        if len(self.current_user_processes(user)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

        if not (self.path.joinpath(name)).exists():
            raise Exception("The model does not exist")

        # load the model
        b = BertModel(name, self.path / name)
        b.load(lazy=True)
        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictBert(
                df=df,
                col_text=col_text,
                col_label=col_label,
                path=b.path,
                basemodel=b.base_model,
                file_name=f"predict_{dataset}.parquet",
                batch=batch_size,
            ),
        )
        b.status = f"predicting {dataset}"
        self.computing.append(
            UserModelComputing(
                user=user,
                model=b,
                model_name=b.name,
                unique_id=unique_id,
                time=datetime.now(),
                kind="predict_bert",
                dataset=dataset,
                status="predicting",
                get_training_progress=b.get_training_progress,
            )
        )
        return {"success": "bert model predicting"}

    def start_compression(self, name):
        """
        Compress bertmodel as a separate process
        """
        process = Process(
            target=shutil.make_archive,
            args=(self.path / "../../static" / name, "gztar", self.path / name),
        )
        process.start()
        print("starting compression")

    def rename(self, former_name: str, new_name: str):
        """
        Rename a model (copy it)
        """
        # get model
        model = self.projects_service.get_model(self.project_slug, former_name)
        if model is None:
            raise Exception("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.projects_service.rename_model(self.project_slug, former_name, new_name)
        os.rename(model.path, model.path.replace(former_name, new_name))
        return {"success": "model renamed"}

    def get(self, name: str, lazy=False) -> BertModel | None:
        """
        Get a model
        """
        model = self.projects_service.get_model(self.project_slug, name)
        if model is None:
            return None
        if not Path(model.path).exists():
            return None
        if (Path(model.path) / "status.log").exists():
            return None
        b = BertModel(name, Path(model.path))
        b.load(lazy=lazy)
        return b

    def export_prediction(
        self, name: str, file_name: str = "predict.parquet", format: str | None = None
    ):
        """
        Export predict file if exists
        """
        path = self.path / name / file_name

        # change format if needed
        if format == "csv":
            df = pd.read_parquet(path)
            file_name = "predict.csv"
            path = self.path / name / file_name
            df.to_csv(path)
        # change format if needed
        if format == "xlsx":
            df = pd.read_parquet(path)
            file_name = "predict.xlsx"
            path = self.path / name / file_name
            df.to_excel(path)

        if not path.exists():
            raise FileNotFoundError("file does not exist")

        r = {"name": file_name, "path": path}
        return r

    def export_bert(self, name: str) -> StaticFileModel:
        """
        Export bert archive if exists
        """
        file_name = f"{name}.tar.gz"
        if not (self.path / "../../static" / file_name).exists():
            raise FileNotFoundError("file does not exist")
        return StaticFileModel(
            name=file_name,
            path=str(self.path.joinpath("../../static").joinpath(file_name)),
        )

    def add(self, element: UserModelComputing):
        """
        Manage computed process for model
        """
        if element.status == "training":
            # update bdd status
            self.projects_service.change_model_status(
                self.project_slug, element.model_name, "trained"
            )
            print("Model trained")
        if element.status == "testing":
            print("Model tested")
        if element.status == "predicting":
            # case of global prediction completed
            if element.dataset == "all":
                self.projects_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted",
                    value=True,
                )
            print("Prediction finished")


class SimpleModels:
    """
    Managing simplemodels
    - define available models
    - save a simplemodel/user
    - train simplemodels
    """

    available_models: dict
    validation: dict
    existing: dict
    computing: list
    path: Path
    queue: Queue
    save_file: str

    def __init__(self, path: Path, queue: Any, computing: list) -> None:
        """
        Init Simplemodels class
        """
        # Models and default parameters
        self.available_models = {
            "liblinear": {"cost": 1},
            "knn": {"n_neighbors": 3},
            "randomforest": {"n_estimators": 500, "max_features": None},
            "lasso": {"C": 32},
            "multi_naivebayes": {"alpha": 1, "fit_prior": True, "class_prior": None},
        }

        # To validate JSON
        self.validation = {
            "liblinear": LiblinearParams,
            "knn": KnnParams,
            "randomforest": RandomforestParams,
            "lasso": LassoParams,
            "multi_naivebayes": Multi_naivebayesParams,
        }
        self.existing: dict = {}  # computed simplemodels
        self.computing: list = computing  # currently under computation
        self.path: Path = path  # path to operate
        self.queue = queue  # access to executor for multiprocessing
        self.save_file: str = "simplemodels.pickle"  # file to save current state
        self.loads()  # load existing simplemodels

    def __repr__(self) -> str:
        return str(self.available())

    def available(self):
        """
        Available simplemodels
        """
        r = {}
        for u in self.existing:
            r[u] = {}
            for s in self.existing[u]:
                sm = self.existing[u][s]
                r[u][s] = {
                    "model": sm.name,
                    "params": sm.model_params,
                    "features": sm.features,
                    "statistics": sm.statistics,
                }
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
                    statistics_cv10=sm.cv10,
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
        sm = self.existing[username][scheme]
        return sm.proba

    def training(self) -> dict:
        """
        Currently under training
        """
        r = {e.user: list(e.scheme) for e in self.computing if e.kind == "simplemodel"}
        return r

    def exists(self, user: str, scheme: str):
        """
        Test if a simplemodel exists for a user/scheme
        """
        if user in self.existing:
            if scheme in self.existing[user]:
                return True
        return False

    def get_model(self, user: str, scheme: str):
        """
        Select a specific model in the repo
        """
        if user not in self.existing:
            return "This user has no model"
        if scheme not in self.existing[user]:
            return "The model for this scheme does not exist"
        return self.existing[user][scheme]

    def load_data(self, data, col_label, col_predictors, standardize):
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

    def standardize(self, df):
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
    ):
        """
        Add a new simplemodel for a user and a scheme
        """
        logger_simplemodel = logging.getLogger("simplemodel")
        logger_simplemodel.info("Intiating the computation process for the simplemodel")
        X, Y, labels = self.load_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[name]

        # Select model
        if name == "knn":
            model = KNeighborsClassifier(
                n_neighbors=int(model_params["n_neighbors"]), n_jobs=-1
            )

        if name == "lasso":
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=model_params["C"], n_jobs=-1
            )

        if name == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", C=model_params["cost"], n_jobs=-1
            )

        if name == "randomforest":
            # params  Num. trees mtry  Sample fraction
            # Number of variables randomly sampled as candidates at each split:
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            model = RandomForestClassifier(
                n_estimators=int(model_params["n_estimators"]),
                random_state=42,
                max_features=(
                    int(model_params["max_features"])
                    if model_params["max_features"]
                    else None
                ),
                n_jobs=-1,
            )

        if name == "multi_naivebayes":
            # small workaround for parameters
            if "class_prior" in model_params and model_params["class_prior"]:
                class_prior = model_params["class_prior"]
            else:
                class_prior = None
            # Only with dtf or tfidf for features
            # TODO: calculate class prior for docfreq & termfreq
            model = MultinomialNB(
                alpha=model_params["alpha"],
                fit_prior=model_params["fit_prior"],
                class_prior=class_prior,
            )

        # launch the compuation (model + statistics) as a future process
        args = {"model": model, "X": X, "Y": Y, "labels": labels}
        unique_id = self.queue.add_task("simplemodel", project_slug, FitModel(**args))
        del args

        sm = SimpleModel(
            name, user, X, Y, labels, "computing", features, standardize, model_params
        )
        self.computing.append(
            UserModelComputing(
                user=user,
                model=sm,
                model_name=sm.name,
                unique_id=unique_id,
                time=datetime.now(),
                kind="simplemodel",
                status="training",
                scheme=scheme,
            )
        )

    def dumps(self):
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

    def add(self, element: UserModelComputing, results) -> None:
        """
        Add simplemodel after computation
        """
        sm = element.model
        sm.model = results.model
        sm.proba = results.proba
        sm.cv10 = results.cv10
        sm.statistics = results.statistics
        if element.user not in self.existing:
            self.existing[element.user] = {}
        self.existing[element.user][element.scheme] = sm
        self.dumps()

    def export_prediction(self, scheme: str, username: str, format: str = "csv"):
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


class SimpleModel:
    name: str
    user: str
    features: list
    X: DataFrame
    Y: DataFrame
    labels: list
    model_params: dict
    standardize: bool
    proba: DataFrame | None
    statistics: MLStatisticsModel | None
    cv10: MLStatisticsModel | None
    # model

    def __init__(
        self,
        name: str,
        user: str,
        X: DataFrame,
        Y: DataFrame,
        labels: list,
        model,
        features: list,
        standardize: bool,
        model_params: dict | None,
    ) -> None:
        """
        Define a specific Simplemodel with parameters
        TODO : add timestamp ?
        TODO : not sure that statistics function are still usefull since it is calculated during the fit
        """
        self.name = name
        self.user = user
        self.features = features
        self.X = X
        self.Y = Y
        self.labels = labels
        self.model_params = model_params
        self.standardize = standardize
        self.model = model
        self.proba = None
        self.statistics = None
        self.cv10 = None

    def json(self):
        """
        Return json representation
        """
        return {
            "name": str(self.name),
            "features": list(self.features),
            "labels": list(self.labels),
            "params": dict(self.model_params),
        }
