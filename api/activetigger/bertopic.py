import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from activetigger.datamodels import (
    BertTopicComputing,
    BertTopicParamsModel,
    BertTopicProjectStateModel,
)
from activetigger.features import Features
from activetigger.queue import Queue
from activetigger.tasks.compute_bertopic import ComputeBertTopic

# TODO : Implement the get_topics and get_projection methods
# TODO : Richer state with defined typemodels


class BertTopic:
    """
    Class to handle BERTopic computations.
    """

    def __init__(
        self, project_slug: str, path: Path, queue: Queue, computing: list, features: Features
    ) -> None:
        self.project_slug = project_slug
        self.queue = queue
        self.computing = computing
        self.path: Path = Path(path).joinpath("bertopic")
        self.path.mkdir(parents=True, exist_ok=True)
        self.features = features

    def compute(
        self,
        path_data: Path,
        col_id: str,
        col_text: str,
        parameters: BertTopicParamsModel,
        name: str,
        user: str,
    ) -> None:
        """
        Compute BERTopic model.
        """

        if len(self.current_user_processes(user)) > 0:
            raise ValueError("You already have computation in progress.")

        args = {
            "path_bertopic": self.path,
            "path_data": path_data,
            "col_id": col_id,
            "col_text": col_text,
            "parameters": parameters,
            "name": name,
        }
        unique_id = self.queue.add_task(
            "bertopic", self.project_slug, ComputeBertTopic(**args), queue="gpu"
        )
        self.computing.append(
            BertTopicComputing(
                user=user,
                unique_id=unique_id,
                name=name,
                path_data=path_data,
                col_id=col_id,
                col_text=col_text,
                parameters=parameters,
                time=datetime.now(),
                kind="bertopic",
                get_progress=self.get_progress(name),
            )
        )

    def training(self) -> dict[str, str]:
        """
        Get available BERTopic models in the current process
        """
        return {
            e.user: e.get_progress() if e.get_progress else "Computing"
            for e in self.computing
            if e.kind == "bertopic"
        }

    def available(self) -> dict[str, str]:
        """
        Get available BERTopic models.
        """
        return {
            i: i
            for i in os.listdir(self.path.joinpath("runs"))
            if (self.path.joinpath("runs") / i).is_dir()
        }

    def state(self) -> BertTopicProjectStateModel:
        return BertTopicProjectStateModel(
            available=self.available(),
            training=self.training(),
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
                print("AVANCEMENT", path_progress.read_text())
                return path_progress.read_text()
            return None

        return progress

    def get_topics(self, name: str) -> list:
        pass

    def get_projection(self, name: str) -> list:
        pass
