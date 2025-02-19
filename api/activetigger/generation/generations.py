import pandas as pd
from pandas import DataFrame

from activetigger.datamodels import (
    UserGenerationComputing,
)
from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager


class Generations:
    """
    Class to manage generation data
    """

    computing: list[UserGenerationComputing]
    generations_service: GenerationsService

    def __init__(
        self, db_manager: DatabaseManager, computing: list[UserGenerationComputing]
    ) -> None:
        self.generations_service = db_manager.generations_service
        self.computing = computing

    def add(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        model_id: int,
        prompt: str,
        answer: str,
    ) -> None:
        """
        Add a generated element in the database
        """
        self.generations_service.add_generated(
            user=user,
            project_slug=project_slug,
            element_id=element_id,
            model_id=model_id,
            prompt=prompt,
            answer=answer,
        )
        return None

    def get_generated(
        self,
        project_slug: str,
        username: str,
        n_elements: int,
    ) -> DataFrame:
        """
        Get generated elements from the database
        """
        result = self.generations_service.get_generated(
            project_slug=project_slug, username=username, n_elements=n_elements
        )
        df = pd.DataFrame(
            result, columns=["time", "index", "prompt", "answer", "model name"]
        )
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        return df

    def current_users_generating(self) -> list[UserGenerationComputing]:
        return [e for e in self.computing if e.kind == "generation"]
