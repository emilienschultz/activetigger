import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

from pandas import DataFrame

from activetigger.datamodels import (
    ProjectionComputing,
    ProjectionInStrictModel,
    ProjectionOutModel,
    ProjectionsProjectStateModel,
)
from activetigger.queue import Queue
from activetigger.tasks.compute_projection import ComputeProjection


class Projections:
    """
    Manage projections
    """

    # TODO: Transform available type to dict[str, UserProjection]
    path: Path
    available: dict
    options: dict[str, dict[str, Any]]
    computing: list
    queue: Queue

    def __init__(self, path: Path, computing: list, queue: Queue) -> None:
        self.path = path
        self.computing = computing
        self.queue = queue
        self.available = {}
        self.options = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": ["cosine", "euclidean"],
            },
            "tsne": {
                "n_components": 2,
                "learning_rate": "auto",
                "init": "random",
                "perplexity": 3,
            },
        }
        self.load()

    def load(self) -> None:
        """
        Load available projections in pickle file
        """
        if self.path.joinpath("projections.pickle").exists():
            try:
                self.available = pickle.load(open(self.path.joinpath("projections.pickle"), "rb"))
            except Exception as e:
                print(e)

    def current_computing(self):
        return [e.name for e in self.computing if e.kind == "projection"]

    def training(self) -> dict[str, str]:
        """
        Currently under training
        """
        r = {e.user: e.method for e in self.computing if e.kind == "projection"}
        return r

    def add(self, element: ProjectionComputing, results: DataFrame) -> None:
        """
        Add projection after computation
        """
        self.available[element.user] = {
            "data": results,
            "method": element.method,
            "parameters": element.params,
            "id": element.unique_id,
        }
        try:
            pickle.dump(self.available, open(self.path.joinpath("projections.pickle"), "wb"))
        except Exception as e:
            print("Error in saving projections", e)

    def compute(
        self,
        project_slug: str,
        username: str,
        projection: ProjectionInStrictModel,
        features: DataFrame,
    ) -> None:
        """
        Launch the projection computation in the queue
        """
        unique_id = self.queue.add_task(
            "projection",
            project_slug,
            ComputeProjection(kind=projection.method, features=features, params=projection.params),
        )
        self.computing.append(
            ProjectionComputing(
                unique_id=unique_id,
                name=f"Projection by {username}",
                user=username,
                time=datetime.now(),
                kind="projection",
                method=projection.method,
                params=projection,
            )
        )

    def get(self, user_name: str) -> ProjectionOutModel | None:
        """
        Get the projection for a user
        """
        if user_name not in self.available:
            return None
        if "data" not in self.available[user_name]:
            return None
        data = self.available[user_name]["data"]
        return ProjectionOutModel(
            index=list(data.index),
            x=list(data[0]),
            y=list(data[1]),
            status=self.available[user_name]["id"],
            parameters=self.available[user_name]["parameters"],
        )

    def state(self) -> ProjectionsProjectStateModel:
        return ProjectionsProjectStateModel(
            options=self.options,
            available={i: self.available[i]["id"] for i in self.available},
            training=self.training(),
        )

    def export(
        self,
        user_name: str,
        format: str = "csv",
    ) -> dict[str, str]:
        """
        Export the projection for a user
        """
        if user_name not in self.available:
            raise Exception("No projection available")
        if "data" not in self.available[user_name]:
            raise Exception("No projection available")
        data = self.available[user_name]["data"]
        file_name = f"projection_{user_name}.{format}"
        if format == "csv":
            data.to_csv(self.path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(self.path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(self.path.joinpath(file_name))

        return {"name": file_name, "path": str(self.path.joinpath(file_name))}
