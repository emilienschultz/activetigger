import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
from fastapi.responses import FileResponse
from slugify import slugify

from activetigger.datamodels import (
    BertopicComputing,
    BertopicOutModelParameters,
    BertopicParamsModel,
    BertopicProjectStateModel,
    TopicsOutModel,
)
from activetigger.db.languagemodels import ModelsService
from activetigger.db.manager import DatabaseManager
from activetigger.features import Features
from activetigger.queue import Queue
from activetigger.tasks.compute_bertopic import ComputeBertopic

# TODO : Implement the get_topics and get_projection methods
# TODO : Richer state with defined typemodels


class Bertopic:
    """
    Class to handle BERTopic computations.
    """

    models_service: ModelsService

    def __init__(
        self,
        project_slug: str,
        path: Path,
        queue: Queue,
        computing: list,
        features: Features,
        db_manager: DatabaseManager,
    ) -> None:
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.path: Path = path.joinpath("bertopic")
        self.path.mkdir(parents=True, exist_ok=True)
        self.features = features
        self.available_models = [
            "jinaai/jina-embeddings-v3",
            "multi-qa-mpnet-base-dot-v1",
            "Alibaba-NLP/gte-multilingual-base",
            "all-mpnet-base-v2",
            "all-MiniLM-L6-v2",
            "paraphrase-multilingual-mpnet-base-v2",
        ]
        self.models_service = db_manager.language_models_service

    def compute(
        self,
        path_data: Path,
        col_id: str | None,
        col_text: str,
        parameters: BertopicParamsModel,
        name: str,
        user: str,
        force_compute_embeddings: bool = False,
    ) -> str:
        """
        Compute BERTopic model.
        """

        name = slugify(name)

        if len(self.current_user_processes(user)) > 0:
            raise ValueError("You already have computation in progress.")

        args = ComputeBertopic(
            path_bertopic=self.path,
            path_data=path_data,
            col_id=col_id,
            col_text=col_text,
            parameters=parameters,
            name=name,
            force_compute_embeddings=force_compute_embeddings,
        )
        unique_id = self.queue.add_task("bertopic", self.project_slug, args, queue="gpu")
        self.computing.append(
            BertopicComputing(
                user=user,
                unique_id=unique_id,
                name=name,
                path_data=path_data,
                col_id=col_id,
                col_text=col_text,
                parameters=parameters,
                time=datetime.now(),
                kind="bertopic",
                force_compute_embeddings=force_compute_embeddings,
                get_progress=self.get_progress(name),
            )
        )
        return unique_id

    # def add(self, element: BertopicComputing) -> None:
    #     """
    #     Add a trained BERTopic in the database
    #     """
    #     model_path = self.path.joinpath("runs").joinpath(element.name)
    #     self.models_service.add_model(
    #         kind="bertopic",
    #         name=element.name,
    #         user=element.user,
    #         project=self.project_slug,
    #         scheme="all",
    #         params=element.parameters.model_dump(),
    #         path=str(model_path),
    #         status="computed",
    #     )

    def training(self) -> dict[str, dict[str, str | int | None]]:
        """
        Get available BERTopic models in the current process
        """
        return {
            e.user: {"progress": e.get_progress() or "Computing"}
            for e in self.computing
            if e.kind == "bertopic"
        }

    def available(self) -> dict[str, str | dict[str, str] | None]:
        """
        Get available BERTopic models.
        """

        def retrieve_date(folder_path):
            with open(folder_path / "params.json", "r") as file:
                p = json.load(file)
            return p["timestamp"]

        if self.path.exists() and self.path.joinpath("runs").exists():
            return {
                folder_name: {
                    "name": folder_name,
                    "time": retrieve_date(self.path.joinpath("runs") / folder_name),
                }
                for folder_name in os.listdir(self.path.joinpath("runs"))
                if (self.path.joinpath("runs") / folder_name).is_dir()
                and (self.path.joinpath("runs") / folder_name / "bertopic_topics.csv").exists()
            }
        return {}

    def name_available(self, name: str) -> bool:
        """
        Check if a BERTopic model name is available.
        """
        return slugify(name) not in self.available()

    def state(self) -> BertopicProjectStateModel:
        return BertopicProjectStateModel(
            available=self.available(),
            training=self.training(),  # type: ignore
            models=self.available_models,
        )

    def current_user_processes(self, user: str) -> list:
        """
        Get current user processes
        """
        return [e for e in self.computing if e.user == user]

    def get_progress(self, name) -> Callable[[], Optional[str]]:
        """
        Access the log progess
        """
        path_progress = self.path.joinpath("runs").joinpath(name).joinpath("progress")

        def progress():
            if path_progress.exists():
                return path_progress.read_text()
            return None

        return progress

    def delete(self, name: str) -> None:
        """
        Delete a BERTopic model.
        """

        # on disk
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            shutil.rmtree(path_model)
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def get_topics(self, name: str) -> list[TopicsOutModel]:
        """
        Get topics from a BERTopic model.
        Return a list of dictionaries where a dictionary is a row in the dataframe
        (alike TopicsOutModel).
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            df = pd.read_csv(path_model.joinpath("bertopic_topics.csv"), index_col=0)
            df.columns = df.columns.astype(str)
            df_list = df.to_dict(orient="records")
            return [TopicsOutModel(**item) for item in df_list]  # type: ignore
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def get_clusters(self, name: str) -> dict[str, int]:
        """
        Get clusters from a BERTopic model.
        Return a list of dictionaries where a dictionary is a row in the dataframe
        (structure: {'id' : cluster}).
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            return pd.read_csv(path_model.joinpath("bertopic_clusters.csv"), index_col=0).to_dict()[
                "cluster"
            ]
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def get_parameters(self, name: str) -> BertopicOutModelParameters:
        """
        Get parameters file from a BERTopic model
        TODO : cache ?
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if not path_model.exists():
            raise FileNotFoundError(f"Model {name} does not exist.")
        params_path = path_model.joinpath("params.json")
        if not params_path.exists():
            raise FileNotFoundError(f"Parameters for model {name} do not exist.")
        with open(params_path) as f:
            r = json.load(f)
            return BertopicOutModelParameters(**r)

    def get_projection(self, name: str) -> dict[str, list | dict]:
        """
        Open the project and the cluster
        """
        path_clusters = self.path.joinpath("runs").joinpath(name).joinpath("bertopic_clusters.csv")
        path_projection = self.path.joinpath("runs").joinpath(name).joinpath("projection2D.parquet")
        if not path_clusters.exists() or not path_projection.exists():
            raise FileNotFoundError(f"Projection for model {name} does not exist.")
        clusters = pd.read_csv(path_clusters, index_col=0)
        clusters.index = clusters.index.astype(str)
        projection = pd.read_parquet(path_projection)
        projection["cluster"] = clusters["cluster"]
        path_model = self.path.joinpath("runs").joinpath(name)
        labels = {}
        if path_model.exists():
            df = pd.read_csv(path_model.joinpath("bertopic_topics.csv"), index_col=0)
            labels = dict(df[["Topic", "Name"]].set_index("Topic")["Name"])
        return {
            "x": projection["x"].tolist(),
            "y": projection["y"].tolist(),
            "cluster": projection["cluster"].tolist(),
            "id": projection.index.astype(str).tolist(),
            "labels": labels,
        }

    def export_topics(self, name: str) -> FileResponse:
        """
        Export topics from a BERTopic model.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            topics_path = path_model.joinpath("bertopic_topics.csv")
            if topics_path.exists():
                return FileResponse(path=topics_path, filename=f"bertopic_topics_{name}.csv")
            else:
                raise FileNotFoundError(f"Topics for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def export_clusters(self, name: str) -> FileResponse:
        """
        Export clusters from a BERTopic model.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            clusters_path = path_model.joinpath("bertopic_clusters.csv")
            if clusters_path.exists():
                return FileResponse(path=clusters_path, filename=f"bertopic_clusters_{name}.csv")
            else:
                raise FileNotFoundError(f"Clusters for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")

    def export_report(self, name: str) -> FileResponse:
        """
        Export clusters from a BERTopic model.
        """
        path_model = self.path.joinpath("runs").joinpath(name)
        if path_model.exists():
            report_path = path_model.joinpath("report.html")
            if report_path.exists():
                return FileResponse(path=report_path, filename=f"bertopic_report_{name}.html")
            else:
                raise FileNotFoundError(f"Report for model {name} do not exist.")
        else:
            raise FileNotFoundError(f"Model {name} does not exist.")
