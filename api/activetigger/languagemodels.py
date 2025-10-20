import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, cast

import pandas as pd  # type: ignore[import]
from fastapi.responses import FileResponse
from pandas import DataFrame

import activetigger.functions as functions
from activetigger.config import config
from activetigger.datamodels import (
    LanguageModelsProjectStateModel,
    LMComputing,
    LMComputingOutModel,
    LMInformationsModel,
    LMParametersDbModel,
    LMParametersModel,
    LMStatusModel,
    StaticFileModel,
)
from activetigger.db.languagemodels import LanguageModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.functions import get_scores_prediction
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

    def available(self) -> dict[str, dict[str, LMStatusModel]]:
        """
        Available models
        """
        models = self.language_models_service.available_models(self.project_slug, "bert")
        r: dict = {}
        for m in models:
            if m.scheme not in r:
                r[m.scheme] = {}
            r[m.scheme][m.name] = LMStatusModel(
                predicted=m.parameters.get("predicted", False),
                tested=m.parameters.get("tested", False),
                predicted_external=m.parameters.get("predicted_external", False),
            )
        return r

    def training(self) -> dict[str, LMComputingOutModel]:
        """
        Currently under training
        - name
        - progress if available
        - loss if available
        """

        r = {
            e.user: LMComputingOutModel(
                name=e.model_name,
                status=e.status,
                progress=e.get_progress() if e.get_progress else None,
                loss=self.get_loss(e.model_name),
                epochs=e.params["epochs"] if e.params else None,
            )
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
        loss: str = "cross_entropy",
    ) -> None:
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
            if self.estimate_memory_use(model_name, kind="train") > mem.available_memory:
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
                loss=loss,
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

    def clean_files_valid(self, model_name: str, dataset: str):
        """
        Clean previous files for validation or test dataset
        #TODO : CHECK STATISTICS
        """
        if dataset not in ["valid", "test"]:
            raise Exception("Dataset should be 'valid' or 'test'")
        if (self.path.joinpath(model_name).joinpath(f"predict_{dataset}.parquet")).exists():
            os.remove(self.path.joinpath(model_name).joinpath(f"predict_{dataset}.parquet"))
        if (self.path.joinpath(model_name).joinpath(f"metrics_{dataset}.json")).exists():
            os.remove(self.path.joinpath(model_name).joinpath(f"metrics_{dataset}.json"))

    def start_predicting_process(
        self,
        project_slug: str,
        name: str,
        user: str,
        df: DataFrame,
        dataset: str,
        col_text: str,
        col_label: str | None = None,
        col_id: str | None = None,
        col_datasets: str | None = None,
        batch_size: int = 32,
        status: str = "predicting",
        statistics: list | None = None,
    ) -> None:
        """
        Start predicting process
        """
        if not (self.path.joinpath(name)).exists():
            raise Exception("The model does not exist")

        file_name = f"predict_{dataset}.parquet"

        # load the model
        unique_id = self.queue.add_task(
            "prediction",
            project_slug,
            PredictBert(
                path=self.path.joinpath(name),
                df=df,
                col_text=col_text,
                col_label=col_label,
                col_id=col_id,
                col_datasets=col_datasets,
                basemodel=self.get_base_model(name),
                file_name=file_name,
                batch=batch_size,
                statistics=statistics,
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
                status=status,
                get_progress=self.get_progress(name, status=status),
            )
        )

    def rename(self, former_name: str, new_name: str) -> None:
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

    def export_prediction(
        self, name: str, file_name: str = "predict.parquet", format: str = "parquet"
    ) -> FileResponse:
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

        return FileResponse(
            path=path,
            filename=file_name,
        )

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

    def add(self, element: LMComputing) -> None:
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

    # def get_scores_prediction(self, model_name, dataset) -> dict | None:
    #     """
    #     Get the scores of the model for a dataset
    #     - last metrics file
    #     - return None if not
    #     """
    #     # case for internalvalid
    #     if dataset == "internalvalid":
    #         with open(
    #             self.path.joinpath(model_name).joinpath("metrics_internalvalid.json"),
    #             "r",
    #         ) as f:
    #             valid_scores = json.load(f)
    #         return valid_scores

    #     folder = self.path.joinpath(model_name)
    #     files = sorted(
    #         [
    #             f.name
    #             for f in folder.iterdir()
    #             if f.is_file() and f.name.startswith("metrics_predict_")
    #         ],
    #     )
    #     if len(files) == 0:
    #         return None
    #     last_stat_file = files[-1]
    #     with open(folder.joinpath(last_stat_file), "r") as f:
    #         stats = json.load(f)
    #     if dataset in stats:
    #         return stats[dataset]
    #     else:
    #        return None

    def get_informations(self, model_name) -> LMInformationsModel:
        """
        Informations on the bert model from the files
        TODO : avoid to read and create a cache
        """

        loss = self.get_loss(model_name)
        params = self.get_parameters(model_name)
        internalvalid_scores = get_scores_prediction(
            self.path.joinpath(model_name),
            "internalvalid",
        )
        train_scores = get_scores_prediction(self.path.joinpath(model_name), "train")
        valid_scores = get_scores_prediction(self.path.joinpath(model_name), "valid")
        test_scores = get_scores_prediction(self.path.joinpath(model_name), "test")
        # outofsample_scores = get_scores_prediction(self.path.joinpath(model_name), "outofsample")
        outofsample_scores = None

        return LMInformationsModel(
            params=params,
            loss=loss,
            train_scores=train_scores,
            internalvalid_scores=internalvalid_scores,
            valid_scores=valid_scores,
            test_scores=test_scores,
            outofsample_scores=outofsample_scores,
        )

    def get_base_model(self, model_name) -> dict:
        """
        Get the base model for a model
        """
        with open(self.path.joinpath(model_name).joinpath("parameters.json"), "r") as jsonfile:
            data = json.load(jsonfile)
            if "base_model" in data:
                return data["base_model"]
            else:
                raise ValueError("No model type found in config.json. Please check the file.")

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

        return [str(i) for i in pd.read_csv(path, index_col=0).index]

    def get_train_ids(self, model_name: str) -> list[str]:
        """
        Get the training ids from the train dataset of the model
        """
        path = self.path.joinpath(model_name).joinpath("train_dataset_eval.csv")
        if not path.exists():
            raise FileNotFoundError("Training ids file does not exist")
        return [str(i) for i in pd.read_csv(path, index_col=0).index]
