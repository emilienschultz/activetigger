import logging
from pathlib import Path
from typing import cast

from pandas import DataFrame

from activetigger.datamodels import (
    GenerationModel,
    GenerationResult,
)
from activetigger.generation.client import GenerationModelClient
from activetigger.generation.huggingface import HuggingFace
from activetigger.generation.ollama import Ollama
from activetigger.generation.openai import OpenAI
from activetigger.generation.openrouter import OpenRouter
from activetigger.tasks.base_task import BaseTask


class GenerateCall(BaseTask):
    """
    Generate call for api
    """

    kind = "generate_call"

    def __init__(
        self,
        path_process: Path | None,
        username: str,
        project_slug: str,
        df: DataFrame,
        model: GenerationModel,
        prompt: str,
    ):
        super().__init__()
        if path_process is None:
            path_process = Path(".")
        self.path_process = path_process
        self.username = username
        self.project_slug = project_slug
        self.df = df
        self.model = model
        self.prompt = prompt

    def _write_progress(self, progress: int):
        """
        Write progress in the file
        """
        with open(self.path_process.joinpath(self.unique_id), "w") as f:
            f.write(f"{progress}")
        print(f"Progress: {progress}")

    @staticmethod
    def get_progress_callback(path_file):
        """
        Get progress callback
        """

        def callback() -> int | None:
            try:
                with open(path_file, "r") as f:
                    r = f.read()
                return int(r)
            except Exception as e:
                print(e)
                return None

        return callback

    def __call__(self):
        """
        Generate call for api
        """
        """
        Manage batch generation request
        Return table of results
        """
        # errors
        errors: list[Exception] = []
        results: list[GenerationResult] = []

        # loop on all elements
        # TODO: Why not give all the data in one go?
        c = 0
        self._write_progress(0)
        for _index, row in self.df.iterrows():
            # test for interruption
            if self.event is not None:
                if self.event.is_set():
                    raise Exception("Process was interrupted")

            # insert the content in the prompt (either at the end or where it is indicated)
            if "#INSERTTEXT" in self.prompt:
                prompt_with_text = self.prompt.replace(
                    "#INSERTTEXT", cast(str, row["text"])
                )
            else:
                prompt_with_text = self.prompt + "\n\n" + cast(str, row["text"])

            # Get configured model

            # make request to the client
            gen_model: GenerationModelClient
            if self.model.api == "Ollama":
                if self.model.endpoint is None:
                    raise Exception("You need to give an endpoint for the Ollama model")
                gen_model = Ollama(self.model.endpoint)
            elif self.model.api == "OpenAI":
                if self.model.credentials is None:
                    raise Exception(
                        "You need to give your API key to call an OpenAI model"
                    )
                gen_model = OpenAI(self.model.credentials)
            elif self.model.api == "HuggingFace":
                gen_model = HuggingFace(
                    credentials=self.model.credentials, endpoint=self.model.endpoint
                )
            elif self.model.api == "OpenRouter":
                gen_model = OpenRouter(credentials=self.model.credentials)
            else:
                errors.append(Exception("Model does not exist"))
                continue

            response = gen_model.generate(prompt_with_text, self.model.slug)
            results.append(
                GenerationResult(
                    user=self.username,
                    project_slug=self.project_slug,
                    model_id=self.model.id,
                    element_id=_index,
                    prompt=prompt_with_text,
                    answer=response,
                )
            )
            logging.debug("element generated: %s, %s ", _index, response)
            self._write_progress(int((c / len(self.df)) * 100))
            c += 1

        return results
