import multiprocessing
from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Optional

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class BaseTask(ABC):
    def __init__(self, **kwargs):
        """Initialize the task with given parameters."""
        self.params = kwargs

    @abstractmethod
    def process(self):
        """The function that realize the task processing."""
        pass

    @abstractmethod
    def clean(self, result):
        """The function to call when the task is completed to clean"""
        pass


class CustomLoggingCallback(TrainerCallback):
    event: Optional[multiprocessing.synchronize.Event]
    current_path: Path
    logger: Logger

    def __init__(self, event, logger, current_path):
        self.event = event
        self.current_path = current_path
        self.logger = logger

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.logger.info(f"Step {state.global_step}")
        progress_percentage = (state.global_step / state.max_steps) * 100
        with open(self.current_path.joinpath("train/progress"), "w") as f:
            f.write(str(progress_percentage))
        # end if event set
        if self.event is not None:
            if self.event.is_set():
                self.logger.info("Event set, stopping training.")
                control.should_training_stop = True
                raise Exception("Process interrupted by user")
