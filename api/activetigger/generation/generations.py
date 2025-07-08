import pandas as pd
from pandas import DataFrame

from activetigger.datamodels import (  # ignore[import]
    GenerationComputing,
    GenerationComputingOut,
    GenerationCreationModel,
    GenerationsProjectStateModel,
    PromptModel,
)
from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager
from activetigger.functions import remove_punctuation, replace_accented_chars


class Generations:
    """
    Class to manage generation data
    """

    computing: list
    generations_service: GenerationsService

    def __init__(self, db_manager: DatabaseManager, computing: list[GenerationComputing]) -> None:
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
            user_name=user,
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
        user_name: str,
        n_elements: int | None = None,
    ) -> DataFrame:
        """
        Get generated elements from the database
        """
        result = self.generations_service.get_generated(
            project_slug=project_slug, user_name=user_name, n_elements=n_elements
        )
        df = pd.DataFrame(result, columns=["time", "index", "prompt", "answer", "model name"])
        df["time"] = pd.to_datetime(df["time"])
        if df["time"].dt.tz is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        return df

    def training(self) -> dict[str, GenerationComputingOut]:
        """
        Get state current generation computing
        """
        return {
            e.user: GenerationComputingOut(
                model_id=e.model_id,
                progress=e.get_progress() if e.get_progress is not None else 0,
            )
            for e in self.computing
            if e.kind == "generation"
        }

    def prompt_exists(self, project_slug: str, name: str) -> bool:
        """
        Check if a prompt already exists
        """
        all_prompts = self.get_prompts(project_slug)
        return any([prompt.parameters["name"] == name for prompt in all_prompts])

    def model_exists(self, project_slug: str, name: str) -> bool:
        """
        Check if a model already exists
        """
        all_models = self.generations_service.get_project_gen_models(project_slug)
        return any([model.name == name for model in all_models])

    def add_model(self, project_slug: str, model: GenerationCreationModel, user_name: str) -> int:
        """
        Add a model in the database
        """
        return self.generations_service.add_project_gen_model(project_slug, model, user_name)

    def save_prompt(self, user_name: str, project_slug: str, prompt: str, name: str) -> None:
        """
        Save a prompt in the database
        """
        self.generations_service.add_prompt(
            user_name=user_name,
            project_slug=project_slug,
            text=prompt,
            parameters={"name": name},
        )
        return None

    def delete_prompt(self, prompt_id: int) -> None:
        """
        Delete a prompt from the database
        """
        self.generations_service.delete_prompt(prompt_id)
        return None

    def get_prompts(self, project_slug: str) -> list[PromptModel]:
        """
        Get the list of prompts for the user
        """
        return self.generations_service.get_prompts(project_slug)

    def drop_generated(self, project_slug: str, user_name: str) -> None:
        """
        Drop all elements from prediction for a user
        """
        self.generations_service.drop_generated(project_slug, user_name)
        return None

    def filter(self, answers: pd.Series, filters) -> pd.Series:
        """
        Apply filters
        """
        if "remove_punct" in filters:
            answers = answers.apply(remove_punctuation)
        if "remove_spaces" in filters:
            answers = answers.str.replace(r"\s+", " ")
        if "lowercase" in filters:
            answers = answers.str.lower()
        if "strip" in filters:
            answers = answers.str.strip()
        if "replace_accents" in filters:
            answers = answers.apply(replace_accented_chars)
        return answers

    def state(self) -> GenerationsProjectStateModel:
        return GenerationsProjectStateModel(training=self.training())
