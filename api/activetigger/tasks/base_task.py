from abc import ABC, abstractmethod


class BaseTask(ABC):
    def __init__(self, **kwargs):
        """Initialize the task with given parameters."""
        self.params = kwargs
        self.event = None
        self.unique_id = None

    @abstractmethod
    def __call__(self):
        """The function that realize the task processing."""
        pass

    @abstractmethod
    def clean(self, result):
        """The function to call when the task is completed to clean"""
        pass
