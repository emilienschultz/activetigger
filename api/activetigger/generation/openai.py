from openai import OpenAI as OpenAIClient

from activetigger.generation.client import GenerationModelClient


class OpenAI(GenerationModelClient):
    client: OpenAIClient

    def __init__(self, credentials: str):
        self.client = OpenAIClient(api_key=credentials)

    def generate(self, prompt: str, model: str) -> str:
        print("__???", model, prompt)
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
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "annotation_schema",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "input": {
                                "type": "string",
                                "description": "Input to annotate",
                            },
                            "annotation": {
                                "type": "string",
                                "description": "Annotation to categorize the input",
                            },
                            "additional_properties": False,
                        },
                    },
                },
            },
        )
        if response.choices[0].message.content is None:
            raise Exception("ChatGPT could not generate annotations")

        return response.choices[0].message.content
