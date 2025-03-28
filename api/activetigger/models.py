import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore[import]

import activetigger.functions as functions
from activetigger.datamodels import (
    BertModelInformationsModel,
    BertModelParametersDbModel,
    BertModelParametersModel,
    StaticFileModel,
    UserModelComputing,
)
from activetigger.db.manager import DatabaseManager
from activetigger.db.projects import ProjectsService
from activetigger.queue import Queue
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
        print(
            "progress",
            (self.path.joinpath("progress_train")).exists(),
            (self.path.joinpath("progress_train")),
        )
        if (self.status == "training") & (
            self.path.joinpath("progress_train")
        ).exists():
            with open(self.path.joinpath("progress_train"), "r") as f:
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
    queue: Queue
    computing: list[UserModelComputing]
    projects_service: ProjectsService
    db_manager: DatabaseManager
    base_models: list
    params_default: BertModelParametersModel

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
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
                    "name": "flaubert/flaubert_base_cased",
                    "priority": 7,
                    "comment": "",
                    "language": "fr",
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
                "predicted_external": m["parameters"].get("predicted_external", False),
            }
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
            os.remove(
                f"{os.environ['ACTIVETIGGER_PATH']}/static/{self.project_slug}/{bert_name}.tar.gz"
            )
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
        minutes = current_date.strftime("%M")
        hour = current_date.strftime("%H")
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")
        model_name = f"{name}__{user}__{project}__{scheme}__{day}-{month}-{year}_{hour}h{minutes}"

        # check if a project not already exist
        if self.projects_service.model_exists(project, model_name):
            raise Exception("A model with this name already exists")

        # if GPU requested, test if enough memory is available (to avoid CUDA out of memory)
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if (
                self.estimate_memory_use(model_name, kind="train")
                > mem["available_memory"]
            ):
                raise Exception(
                    "Not enough GPU memory available. Wait or reduce batch."
                )

        # launch as a independant process
        unique_id = self.queue.add_task(
            "training",
            project,
            TrainBert(
                path=self.path,
                project_slug=project,
                model_name=model_name,
                df=df.copy(deep=True),
                col_label=col_label,
                col_text=col_text,
                base_model=base_model,
                params=params,
                test_size=test_size,
            ),
            queue="gpu",
        )
        del df

        # Update the queue state
        b = BertModel(model_name, self.path / model_name, base_model)
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
            name=model_name,
            user=user,
            project=project,
            scheme=scheme,
            params=params.model_dump(),
            path=str(self.path / model_name),
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
            queue="gpu",
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
        b = BertModel(name, self.path.joinpath(name))
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
            queue="gpu",
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
        if not path.exists():
            raise FileNotFoundError("file does not exist")

        df = pd.read_parquet(path)

        # change format if needed
        if format == "csv":
            file_name = "predict.csv"
            path = self.path / name / file_name
            df.to_csv(path)
        # change format if needed
        elif format == "xlsx":
            file_name = "predict.xlsx"
            path = self.path / name / file_name
            df.to_excel(path)
        else:
            raise Exception("Format not supported")

        r = {"name": file_name, "path": path}
        return r

    def export_bert(self, name: str) -> StaticFileModel:
        """
        Export bert archive if exists
        """
        file = f"{os.environ['ACTIVETIGGER_PATH']}/static/{self.project_slug}/{name}.tar.gz"

        if not Path(file).exists():
            raise FileNotFoundError("file does not exist")
        return StaticFileModel(
            name=f"{name}.tar.gz",
            path=f"/static/{self.project_slug}/{name}.tar.gz",
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
            # TODO test if the model is already compressed
            self.projects_service.set_model_params(
                self.project_slug,
                element.model_name,
                "compressed",
                True,
            )
            print("Model trained")
        if element.status == "testing":
            print("Model tested")
        if element.status == "predicting":
            # update flag if there is a prediction of the whole dataset
            if element.dataset == "all":
                self.projects_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted",
                    value=True,
                )
            # update flag if there is a prediction in an external dataset
            if element.dataset == "external":
                self.projects_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted_external",
                    value=True,
                )
            print("Prediction finished")
