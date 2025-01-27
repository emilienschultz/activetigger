import logging
from multiprocessing.synchronize import Event
from typing import cast

import pandas as pd
from pandas import DataFrame
from pydantic import BaseModel

from activetigger.datamodels import GenerationModel, UserGenerationComputing
from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager
from activetigger.generation.client import GenerationModelClient
from activetigger.generation.huggingface import HuggingFace
from activetigger.generation.ollama import Ollama
from activetigger.generation.openai import OpenAI


class GenerationResult(BaseModel):
    user: str
    project_slug: str
    model_id: int
    element_id: str
    prompt: str
    answer: str


class Generations:
    """
    Class to manage generation data
    """

    computing: list[UserGenerationComputing]
    generations_service: GenerationsService

    @staticmethod
    def generate(
        user: str,
        project_name: str,
        df: DataFrame,
        model: GenerationModel,
        prompt: str,
        event: Event | None = None,
    ) -> list[GenerationResult]:
        """
        Manage batch generation request
        Return table of results
        """
        # errors
        errors: list[Exception] = []
        results: list[GenerationResult] = []

        # loop on all elements
        # TODO: Why not give all the data in one go?
        for _index, row in df.iterrows():
            # test for interruption
            if event is not None:
                if event.is_set():
                    raise Exception("Process was interrupted")

            # insert the content in the prompt (either at the end or where it is indicated)
            if "#INSERTTEXT" in prompt:
                prompt_with_text = prompt.replace("#INSERTTEXT", cast(str, row["text"]))
            else:
                prompt_with_text = prompt + "\n\n" + cast(str, row["text"])

            # Get configured model

            # make request to the client
            gen_model: GenerationModelClient
            if model.api == "Ollama":
                if model.endpoint is None:
                    raise Exception("You need to give an endpoint for the Ollama model")
                gen_model = Ollama(model.endpoint)
            elif model.api == "OpenAI":
                if model.credentials is None:
                    raise Exception(
                        "You need to give your API key to call an OpenAI model"
                    )
                gen_model = OpenAI(model.credentials)
            elif model.api == "HuggingFace":
                if model.credentials is None:
                    raise Exception(
                        "You need to give your API key to call a HuggingFace model"
                    )
                gen_model = HuggingFace(
                    credentials=model.credentials, endpoint=model.endpoint
                )
            else:
                errors.append(Exception("Model does not exist"))
                continue
            try:
                response = gen_model.generate(prompt_with_text, model.name)
                results.append(
                    GenerationResult(
                        user=user,
                        project_slug=project_name,
                        model_id=model.id,
                        element_id=cast(str, row["id"]),
                        prompt=prompt_with_text,
                        answer=response,
                    )
                )
                logging.debug("element generated: %s, %s ", row["id"], response)
            except Exception as e:
                errors.append(e)

        return results

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
            result, columns=["time", "index", "prompt", "answer", "endpoint"]
        )
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        return df

    def current_users_generating(self) -> list[UserGenerationComputing]:
        print("----+++", [e for e in self.computing])
        return [e for e in self.computing if e.kind == "generation"]
