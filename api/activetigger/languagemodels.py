import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, cast

import pandas as pd
from pandas import DataFrame

import activetigger.functions as functions
from activetigger.config import config
from activetigger.datamodels import (
    LanguageModelsProjectStateModel,
    LMComputing,
    LMInformationsModel,
    LMParametersDbModel,
    LMParametersModel,
    StaticFileModel,
)
from activetigger.db.languagemodels import LanguageModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.queue import Queue
from activetigger.tasks.predict_bert import PredictBert
from activetigger.tasks.train_bert import TrainBert


class LanguageModels:
    """
    Module to manage languagemodels
    """

    project_slug: str
    path: Path
    queue: Queue
    computing: list
    language_models_service: LanguageModelsService
    db_manager: DatabaseManager
    base_models: list[dict[str, Any]]
    params_default: LMParametersModel

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        db_manager: DatabaseManager,
        list_models: str | None = None,
    ) -> None:
        self.params_default = LMParametersModel()
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.language_models_service = db_manager.language_models_service
        self.path: Path = Path(path).joinpath("bert")

        # load the list of models
        if list_models is not None:
            self.base_models = cast(
                list[dict[str, Any]], pd.read_csv(list_models).to_dict(orient="records")
            )

            print(self.base_models)
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

        # create the directory for models
        if not self.path.exists():
            os.mkdir(self.path)

    def __repr__(self) -> str:
        return f"Trained models : {self.available()}"

    def available(self) -> dict[str, dict[str, dict[str, bool]]]:
        """
        Available models
        TODO : change structure ?
        """
        models = self.language_models_service.available_models(self.project_slug)
        r: dict = {}
        for m in models:
            if m.scheme not in r:
                r[m.scheme] = {}
            r[m.scheme][m.name] = {
                "predicted": m.parameters.get("predicted", False),
                "tested": m.parameters.get("tested", False),
                "predicted_external": m.parameters.get("predicted_external", False),
            }
        return r

    def training(self) -> dict[str, dict[str, dict[str, str | None]]]:
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
                "progress": (e.get_progress() if e.get_progress else None),
                "loss": self.get_loss(e.model_name),
                "epochs": e.params["epochs"] if e.params else None,
            }
            for e in self.computing
            if e.kind in ["bert", "train_bert", "predict_bert"]
        }
        return r

    def delete(self, name: str) -> None:
        """
        Delete bert model
        """
        # remove from database
        if not self.language_models_service.delete_model(self.project_slug, name):
            raise FileNotFoundError("Model does not exist")

        # remove files associated
        try:
            if name and name != "":
                shutil.rmtree(self.path.joinpath(name))
                os.remove(f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz")
        except Exception as e:
            raise Exception(f"Problem to delete model files : {e}")

    def current_user_processes(self, user: str) -> list[LMComputing]:
        """
        Get the user current processes
        """
        return [e for e in self.computing if e.user == user]

    def estimate_memory_use(self, model: str, kind: str = "train") -> int:
        """
        Estimate the GPU memory in Gb needed to train a model
        For the moment dummy values
        TODO : implement
        """
        if kind == "train":
            return 4
        if kind == "predict":
            return 3
        return 0

    def start_training_process(
        self,
        name: str,
        project: str,
        user: str,
        scheme: str,
        df: DataFrame,
        col_text: str,
        col_label: str,
        params: LMParametersModel,
        base_model: str = "almanach/camembert-base",
        test_size: float = 0.2,
        num_min_annotations: int = 10,
        num_min_annotations_per_label: int = 5,
    ):
        """
        Manage the training of a model from the API
        """

        # check the size of training data
        if len(df.dropna()) < num_min_annotations:
            raise Exception(f"Less than {num_min_annotations} elements annotated")

        # check the number of elements
        counts = df[col_label].value_counts()
        if not (counts >= num_min_annotations_per_label).all():
            raise Exception(f"Less than {num_min_annotations_per_label} elements per label")

        # name integrating the scheme & user + date
        current_date = datetime.now()
        minutes = current_date.strftime("%M")
        hour = current_date.strftime("%H")
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")
        model_name = f"{name}__{user}__{project}__{scheme}__{day}-{month}-{year}_{hour}h{minutes}"

        # check if a project not already exist
        if self.language_models_service.model_exists(project, model_name):
            raise Exception("A model with this name already exists")

        # if GPU requested, test if enough memory is available (to avoid CUDA out of memory)
        if params.gpu:
            mem = functions.get_gpu_memory_info()
            if self.estimate_memory_use(model_name, kind="train") > mem["available_memory"]:
                raise Exception("Not enough GPU memory available. Wait or reduce batch.")

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

        # add flags in params
        params = LMParametersDbModel(**params.model_dump())

        # Update the queue state
        self.computing.append(
            LMComputing(
                user=user,
                model_name=model_name,
                unique_id=unique_id,
                time=current_date,
                kind="train_bert",
                status="training",
                scheme=scheme,
                dataset=None,
                params=params.model_dump(),
                get_progress=self.get_progress(model_name, status="training"),
            )
        )

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

        # test if the testset and the model have the same labels
        labels_model = self.get_labels(name)
        labels_test = set(df["labels"].dropna().unique())
        if set(labels_model) != set(df["labels"].dropna().unique()):
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
                path=self.path.joinpath(name),
                basemodel=self.get_base_model(name),
                file_name="predict_test.parquet",
                batch=32,
                statistics="full",
            ),
            queue="gpu",
        )

        self.computing.append(
            LMComputing(
                user=user,
                model_name=name,
                unique_id=unique_id,
                time=datetime.now(),
                kind="predict_bert",
                status="testing",
                get_progress=self.get_progress(name, status="predicting"),
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
        col_id: str | None = None,
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
        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictBert(
                df=df,
                col_text=col_text,
                col_label=col_label,
                col_id=col_id,
                path=self.path.joinpath(name),
                basemodel=self.get_base_model(name),
                file_name=f"predict_{dataset}.parquet",
                batch=batch_size,
                statistics="outofsample",
            ),
            queue="gpu",
        )
        self.computing.append(
            LMComputing(
                user=user,
                model_name=name,
                unique_id=unique_id,
                time=datetime.now(),
                kind="predict_bert",
                dataset=dataset,
                status="predicting",
                get_progress=self.get_progress(name, status="predicting"),
            )
        )
        return {"success": "bert model predicting"}

    def rename(self, former_name: str, new_name: str):
        """
        Rename a model (copy it)
        """
        # get model
        model = self.language_models_service.get_model(self.project_slug, former_name)
        if model is None:
            raise Exception("Model does not exist")
        if (Path(model.path) / "status.log").exists():
            raise Exception("Model is currently computing")
        self.language_models_service.rename_model(self.project_slug, former_name, new_name)
        os.rename(model.path, model.path.replace(former_name, new_name))
        return {"success": "model renamed"}

    def export_prediction(
        self, name: str, file_name: str = "predict.parquet", format: str = "parquet"
    ):
        """
        Export predict file if exists
        """
        # get the predition file
        path = self.path.joinpath(name).joinpath(file_name)
        if not path.exists():
            raise FileNotFoundError("file does not exist")
        df = pd.read_parquet(path)

        # add index column
        # read the index column in the parquet file
        # add it in the dataframe

        # change the format
        if format == "parquet":
            pass
        elif format == "csv":
            file_name = file_name + ".csv"
            path = self.path.joinpath(name).joinpath(file_name)
            df.to_csv(path)
        elif format == "xlsx":
            file_name = file_name + ".xlsx"
            path = self.path.joinpath(name).joinpath(file_name)
            df.to_excel(path)
        else:
            raise Exception("Format not supported")

        r = {"name": file_name, "path": path}
        return r

    def export_bert(self, name: str) -> StaticFileModel:
        """
        Export bert archive if exists
        """
        file = f"{config.data_path}/projects/static/{self.project_slug}/{name}.tar.gz"

        if not Path(file).exists():
            raise FileNotFoundError("file does not exist")
        return StaticFileModel(
            name=f"{name}.tar.gz",
            path=f"{self.project_slug}/{name}.tar.gz",
        )

    def add(self, element: LMComputing):
        """
        Manage computed process for model
        """
        if element.status == "training":
            # add in database
            self.language_models_service.add_model(
                kind="bert",
                name=element.model_name,
                user=element.user,
                project=self.project_slug,
                scheme=element.scheme or "default",
                params=element.params or {},
                path=str(self.path.joinpath(element.model_name)),
                status="trained",
            )

            # TODO test if the model is already compressed
            self.language_models_service.set_model_params(
                self.project_slug,
                element.model_name,
                "compressed",
                True,
            )
            print("Model trained")
        if element.status == "testing":
            self.language_models_service.set_model_params(
                self.project_slug,
                element.model_name,
                flag="tested",
                value=True,
            )
            print("Testing finished")
        if element.status == "predicting":
            # update flag if there is a prediction of the whole dataset
            if element.dataset == "all":
                self.language_models_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted",
                    value=True,
                )
            # update flag if there is a prediction in an external dataset
            if element.dataset == "external":
                self.language_models_service.set_model_params(
                    self.project_slug,
                    element.model_name,
                    flag="predicted_external",
                    value=True,
                )
            print("Prediction finished")

    def get_labels(self, model_name: str) -> list:
        """
        Get the labels of the model
        """
        with open(self.path.joinpath(model_name).joinpath("config.json"), "r") as f:
            r = json.load(f)
        return list(r["id2label"].values())

    def get_progress(self, model_name, status: str) -> Callable[[], Optional[float]]:
        """
        Get progress when training
        (different cases)
        """
        if status == "training":
            path_model = self.path.joinpath(model_name).joinpath("progress_train")
        elif status == "predicting":
            path_model = self.path.joinpath(model_name).joinpath("progress_predict")
        else:
            raise Exception("Status not recognized")

        def progress_predicting():
            if path_model.exists():
                r = path_model.read_text()
                if r.strip() == "":
                    r = 0
                return float(r)
            return None

        return progress_predicting

    def get_loss(self, model_name) -> dict | None:
        try:
            with open(self.path.joinpath(model_name).joinpath("log_history.txt"), "r") as f:
                log = json.load(f)
            loss = pd.DataFrame(
                [
                    [
                        log[2 * i]["epoch"],
                        log[2 * i]["loss"],
                        log[2 * i + 1]["eval_loss"],
                    ]
                    for i in range(0, int((len(log)) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            ).to_json()
            return json.loads(loss)
        except Exception:
            return None

    def get_parameters(self, model_name) -> dict | None:
        """
        Get the parameters of the model
        """
        path = self.path.joinpath(model_name).joinpath("parameters.json")

        if not path.exists():
            return None

        with open(path, "r") as jsonfile:
            params = json.load(jsonfile)
        return params

    # def get_trainscores(self, model_name) -> dict | None:
    #     if (self.path.joinpath(model_name).joinpath("metrics_predict_train.parquet.json")).exists():
    #         with open(
    #             self.path.joinpath(model_name).joinpath("metrics_predict_train.parquet.json"),
    #             "r",
    #         ) as f:
    #             train_scores = json.load(f)
    #     else:
    #         train_scores = None
    #     return train_scores

    def get_trainscores(self, model_name) -> dict | None:
        if (self.path.joinpath(model_name).joinpath("metrics_train.json")).exists():
            with open(
                self.path.joinpath(model_name).joinpath("metrics_train.json"),
                "r",
            ) as f:
                train_scores = json.load(f)
        else:
            train_scores = None
        return train_scores

    def get_outofsamplescores(self, model_name) -> dict | None:
        if (self.path.joinpath(model_name).joinpath("metrics_outofsample.json")).exists():
            with open(
                self.path.joinpath(model_name).joinpath("metrics_outofsample.json"),
                "r",
            ) as f:
                outofsample_scores = json.load(f)
        else:
            outofsample_scores = None
        return outofsample_scores

    def get_testscores(self, model_name) -> dict | None:
        if (self.path.joinpath(model_name).joinpath("metrics_predict_test.parquet.json")).exists():
            with open(
                self.path.joinpath(model_name).joinpath("metrics_predict_test.parquet.json"),
                "r",
            ) as f:
                test_scores = json.load(f)
        else:
            test_scores = None
        return test_scores

    def get_validscores(self, model_name) -> dict | None:
        if (self.path.joinpath(model_name).joinpath("metrics_validation.json")).exists():
            with open(
                self.path.joinpath(model_name).joinpath("metrics_validation.json"),
                "r",
            ) as f:
                valid_scores = json.load(f)
        else:
            valid_scores = None
        return valid_scores

    def get_informations(self, model_name) -> LMInformationsModel:
        """
        Informations on the bert model from the files
        TODO : avoid to read and create a cache
        """

        loss = self.get_loss(model_name)
        params = self.get_parameters(model_name)
        valid_scores = self.get_validscores(model_name)
        train_scores = self.get_trainscores(model_name)
        test_scores = self.get_testscores(model_name)
        outofsample_scores = self.get_outofsamplescores(model_name)

        return LMInformationsModel(
            params=params,
            loss=loss,
            train_scores=train_scores,
            test_scores=test_scores,
            valid_scores=valid_scores,
            outofsample_scores=outofsample_scores,
        )

    def get_base_model(self, model_name) -> dict | None:
        """
        Get the base model for a model
        """
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as jsonfile:
            data = json.load(jsonfile)
            if "base_model" in data:
                basemodel = data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")

        return basemodel

    def state(self) -> LanguageModelsProjectStateModel:
        """
        Get the state of the module
        """
        return LanguageModelsProjectStateModel(
            options=self.base_models,
            available=self.available(),
            training=self.training(),
            base_parameters=self.params_default,
        )

    def get_eval_ids(self, model_name: str) -> list[str]:
        """
        Get the evaluation ids from the eval dataset of the model
        """
        path = self.path.joinpath(model_name).joinpath("test_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Evaluation ids file does not exist")

        ids = [str(i) for i in pd.read_csv(path, index_col=0).index]

        return ids

    def get_train_ids(self, model_name: str) -> list[str]:
        """
        Get the training ids from the train dataset of the model
        """
        path = self.path.joinpath(model_name).joinpath("train_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Training ids file does not exist")

        ids = [str(i) for i in pd.read_csv(path, index_col=0).index]

        return ids
