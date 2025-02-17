from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self):
        """Initialize the task with given parameters."""
        self.event = None
        self.unique_id = None

    @abstractmethod
    def __call__(self):
        """The function that realize the task processing."""
        pass
