from abc import ABC, abstractmethod


class GenerationModelClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, model: str | None) -> str:
        pass
