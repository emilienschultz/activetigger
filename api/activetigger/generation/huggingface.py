from huggingface_hub import InferenceClient
from activetigger.generation.client import GenerationModelClient


class HuggingFace(GenerationModelClient):
    endpoint: str
    credentials: str
    client: InferenceClient

    def __init__(self, endpoint: str | None, credentials: str):
        self.endpoint = endpoint or ""
        self.credentials = credentials
        self.client = InferenceClient(token=credentials)

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to a HuggingFace model
        """
        if model is None:
            raise Exception()
        return self.client.chat_completion(prompt, model=self.endpoint + model)
