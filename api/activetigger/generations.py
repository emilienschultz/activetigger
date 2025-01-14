import logging
from multiprocessing.synchronize import Event
from typing import cast

import pandas as pd
import requests
from pandas import DataFrame
from pydantic import BaseModel

from activetigger.db.generations import GenerationsService
from activetigger.db.manager import DatabaseManager


class GenerationResult(BaseModel):
    user: str
    project_slug: str
    endpoint: str
    element_id: str
    prompt: str
    answer: str


class Generations:
    """
    Class to manage generation data
    """

    computing: list
    generations_service: GenerationsService

    @staticmethod
    def generate(
        user: str,
        project_name: str,
        df: DataFrame,
        api: str,
        endpoint: str,
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

            # make request to the client
            if api == "ollama":
                try:
                    response = Generations.request_ollama(endpoint, prompt_with_text)
                    results.append(
                        GenerationResult(
                            user=user,
                            project_slug=project_name,
                            endpoint=endpoint,
                            element_id=cast(str, row["id"]),
                            prompt=prompt_with_text,
                            answer=response,
                        )
                    )
                    logging.debug(
                        "element generated: %s, %s ", row["id"], response["success"]
                    )
                except Exception as e:
                    errors.append(e)
            else:
                errors.append(Exception("Model does not exist"))
                continue

        return results

    @staticmethod
    def request_ollama(endpoint: str, request: str, model: str = "llama3.1:70b"):
        """
        Make a request to ollama
        """
        data = {"model": model, "prompt": request, "stream": False}
        response = requests.post(endpoint, json=data, verify=False)
        print(response.content)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"HTTP call responded with code {response.status_code}")

    def __init__(self, db_manager: DatabaseManager, computing: list) -> None:
        self.generations_service = db_manager.generations_service
        self.computing = computing  # user:{"unique_id", "number","api"}

    def add(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
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
            endpoint=endpoint,
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

    def current_users_generating(self) -> list:
        return {
            e["user"]: {
                "unique_id": e["unique_id"],
                "number": e["number"],
                "api": e["api"],
            }
            for e in self.computing
            if e["kind"] == "generation"
        }

    # def add_generation(
    #     self,
    #     user: str,
    #     project_name: str,
    #     endpoint: str,
    #     element_id: str,
    #     prompt: str,
    #     answer: str,
    # ):
    #     """
    #     Add a generated element in the database
    #     """
    #     self.db_manager.add_generated(
    #         user=user,
    #         project_slug=project_name,
    #         element_id=element_id,
    #         endpoint=endpoint,
    #         prompt=prompt,
    #         answer=answer,
    #     )
    #     print("Added generation", element_id)
    #     return None

    # async def generate(
    #     self, user: str, project_name: str, df: DataFrame, params: GenerateModel
    # ) -> None:
    #     """
    #     Manage generation request
    #     TODO : parallelize
    #     TODO : add other endpoint
    #     """
    #     # errors
    #     errors = []

    #     # loop on all elements
    #     for index, row in df.iterrows():
    #         # insert the content in the prompt
    #         if "#INSERTTEXT" not in params.prompt:
    #             errors.append("Problem with the prompt")
    #             continue
    #         prompt = params.prompt.replace("#INSERTTEXT", row["text"])

    #         # make request to the client
    #         if params.api == "ollama":
    #             response = await functions.request_ollama(params.endpoint, prompt)
    #         else:
    #             errors.append("Model does not exist")
    #             continue

    #         if "error" in response:
    #             errors.append("Error in the request " + response["error"])

    #         if "success" in response:
    #             self.add_generation(
    #                 user,
    #                 project_name,
    #                 params.endpoint,
    #                 row["index"],
    #                 prompt,
    #                 response["success"],
    #             )

    #         print("element generated ", index)

    #     print(errors)
    #     return None

    # async def compute_zeroshot(self, df, params):
    #     """
    #     Zero-shot beta version
    #     # TODO : chunk & control the context size
    #     """
    #     r_error = ["error"] * len(df)

    #     # create the chunks
    #     # FOR THE MOMENT, ONLY 10 elements for DEMO
    #     if len(df) > 10:
    #         df = df[0:10]

    #     # create prompt
    #     list_texts = "\nTexts to annotate:\n"
    #     for i, t in enumerate(list(df["text"])):
    #         list_texts += f"{i}. {t}\n"
    #     prompt = (
    #         params.prompt
    #         + list_texts
    #         + '\nResponse format:\n{"annotations": [{"text": "Text 1", "label": "Label1"}, {"text": "Text 2", "label": "Label2"}, ...]}'
    #     )

    #     # make request to client
    #     # client = openai.OpenAI(api_key=params.token)
    #     try:
    #         self.zeroshot = "computing"
    #         client = openai.AsyncOpenAI(api_key=params.token)
    #         print("Make openai call")
    #         chat_completion = await client.chat.completions.create(
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": """Your are a careful assistant who annotates texts for a research project.
    #                 You follow precisely the guidelines, which can be in different languages.
    #                 """,
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": prompt,
    #                 },
    #             ],
    #             model="gpt-3.5-turbo",
    #             response_format={"type": "json_object"},
    #         )
    #         print("OpenAI call done")
    #     except:
    #         self.zeroshot = None
    #         return {"error": "API connexion failed. Check the token."}
    #     # extracting results
    #     try:
    #         r = json.loads(chat_completion.choices[0].message.content)["annotations"]
    #         r = [i["label"] for i in r]
    #     except:
    #         return {"error": "Format problem"}
    #     if len(r) == len(df):
    #         df["zero_shot"] = r
    #         self.zeroshot = df[["text", "zero_shot"]].reset_index().to_json()
    #         return {"success": "data computed"}
    #     else:
    #         return {"error": "Problem with the number of element"}
