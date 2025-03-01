from datetime import datetime

from pandas import DataFrame

from activetigger.datamodels import ProjectionInStrictModel, UserProjectionComputing
from activetigger.queue import Queue
from activetigger.tasks.compute_projection import ComputeProjection


class Projections:
    """
    Manage projections
    """

    # TODO: Transform available type to dict[str, UserProjection]
    available: dict
    options: dict
    computing: list[UserProjectionComputing]
    queue: Queue

    def __init__(self, computing: list, queue: Queue) -> None:
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

    def current_computing(self):
        return [e.name for e in self.computing if e.kind == "projection"]

    def training(self) -> dict:
        """
        Currently under training
        """
        r = {e.user: e.method for e in self.computing if e.kind == "projection"}
        return r

    def add(self, element: UserProjectionComputing, results: DataFrame) -> None:
        """
        Add projection after computation
        """
        self.available[element.user] = {
            "data": results,
            "method": element.method,
            "params": element.params,
            "id": element.unique_id,
        }

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
            ComputeProjection(
                kind=projection.method, features=features, params=projection.params
            ),
        )
        self.computing.append(
            UserProjectionComputing(
                unique_id=unique_id,
                name=f"Projection by {username}",
                user=username,
                time=datetime.now(),
                kind="projection",
                method=projection.method,
                params=projection,
            )
        )
