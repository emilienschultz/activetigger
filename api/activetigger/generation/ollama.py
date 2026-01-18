import requests  # type: ignore[import]

from activetigger.generation.client import GenerationModelClient


class Ollama(GenerationModelClient):
    endpoint: str

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    def generate(self, prompt: str, model: str) -> str:
        """
        Make a request to ollama
        """
        m = model if model is not None else "llama3.1:70b"
        data = {"model": m, "prompt": prompt, "stream": False}
        response = requests.post(self.endpoint, json=data, verify=False)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"HTTP call responded with code {response.status_code}")
