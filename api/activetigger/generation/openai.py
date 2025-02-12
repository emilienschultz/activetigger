import logging

from openai import OpenAI as OpenAIClient
from openai import RateLimitError

from activetigger.generation.client import GenerationModelClient


class OpenAI(GenerationModelClient):
    client: OpenAIClient

    def __init__(self, credentials: str):
        self.client = OpenAIClient(api_key=credentials)

    def generate(self, prompt: str, model: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "developer",
                        "content": "Your are a careful assistant who annotates texts for a research project in a JSON format. You follow precisely the guidelines, which can be in different languages. ",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        except RateLimitError as rle:
            msg = "Not enough credits on this endpoint"
            logging.error(msg)
            raise Exception(msg) from rle
        except Exception as e:
            msg = getattr(e, "message", repr(e))
            logging.error("Error while calling OpenAI API: %s", e)
            raise Exception("Could not call OpenAI") from e

        if response.choices[0].message.content is None:
            raise Exception("ChatGPT could not generate annotations")

        return response.choices[0].message.content
