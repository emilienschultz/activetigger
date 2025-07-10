####
# Queue management for ActiveTigger
# This module manages the queue of tasks for CPU and GPU processing using Loky.
# There is no multi-GPU support in this version.
####

import asyncio
import datetime
import logging
import multiprocessing
import threading
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager

# manage the executor
from loky import get_reusable_executor  # type: ignore[import]

from activetigger.datamodels import QueueStateTaskModel, QueueTaskModel
from activetigger.tasks.base_task import BaseTask

logger = logging.getLogger("server")
multiprocessing.set_start_method("spawn", force=True)


class Queue:
    """
    Managining parallel processes for computation

    Use Loky as a backend for one common executor

    With 2 waiting queues to differentiate between CPU and GPU jobs
    (to limit concurrency in GPU memory usage)
    """

    max_processes: int = 20
    nb_workers: int
    nb_workers_cpu: int
    nb_workers_gpu: int
    manager: SyncManager
    current: list[QueueTaskModel]
    last_restart: datetime.datetime
    lock: threading.Lock

    def __init__(self, nb_workers_cpu: int = 3, nb_workers_gpu: int = 1) -> None:
        """
        Initiating the queue
        :param nb_workers_cpu: Number of CPU workers
        :param nb_workers_gpu: Number of GPU workers
        :return: None

        Create a Manager to handle shared state across processes
        """
        self.nb_workers_cpu = nb_workers_cpu
        self.nb_workers_gpu = nb_workers_gpu
        self.nb_workers = nb_workers_cpu + nb_workers_gpu
        self.manager = Manager()
        self.current = []
        self.lock = threading.Lock()

        # launch a regular update on the queue
        self.task = asyncio.create_task(self._update_queue(timeout=1))
        logger.info("Init Queue")

    def __del__(self) -> None:
        """
        Destructor to close the queue
        """
        if hasattr(self, "manager"):
            self.manager.shutdown()
        if hasattr(self, "task"):
            self.task.cancel()

    async def _update_queue(self, timeout: int = 1) -> None:
        """
        Update the queue every X seconds.
        Add new tasks to the executor if there are available workers.
        """
        while True:
            with self.lock:  # Ensure thread-safe access to shared state
                # active tasks in the queue
                nb_active_processes_gpu = len(
                    [i for i in self.current if i.queue == "gpu" and i.state == "running"]
                )
                nb_active_processes_cpu = len(
                    [i for i in self.current if i.queue == "cpu" and i.state == "running"]
                )

                # pending tasks in the queue
                task_gpu = [i for i in self.current if i.queue == "gpu" and i.state == "pending"]
                task_cpu = [i for i in self.current if i.queue == "cpu" and i.state == "pending"]

            # get an executor if needed (task to send and available workers)
            if (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers and (
                len(task_gpu) + len(task_cpu) > 0
            ):
                executor = get_reusable_executor(
                    max_workers=(self.nb_workers), timeout=1000, reuse=True
                )

            # a worker available and possible to have gpu
            if (
                nb_active_processes_gpu < self.nb_workers_gpu
                and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
                and len(task_gpu) > 0
            ):
                task_gpu[0].future = executor.submit(task_gpu[0].task)
                task_gpu[0].state = "running"

            # a worker available and possible to have cpu
            if (
                nb_active_processes_cpu < self.nb_workers_cpu
                and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
                and len(task_cpu) > 0
            ):
                task_cpu[0].future = executor.submit(task_cpu[0].task)
                task_cpu[0].state = "running"

            print(
                "nb_active_processes_cpu",
                nb_active_processes_cpu,
                "nb_active_processes_gpu",
                nb_active_processes_gpu,
            )

            await asyncio.sleep(timeout)  # Non-blocking sleep

    def add_task(self, kind: str, project_slug: str, task: BaseTask, queue: str = "cpu") -> str:
        """
        Add a task in the queue, first as pending in the current list
        """
        with self.lock:
            # test if the queue is not full
            if len(self.current) > self.max_processes:
                raise Exception("Queue is full. Wait for process to finish.")

            # generate a unique id
            unique_id = str(uuid.uuid4())
            # set an event to inform the end of the process
            event = self.manager.Event()
            # add informartion in the task
            task.event = event
            task.unique_id = unique_id

            # add it in the current processes
            self.current.append(
                QueueTaskModel(
                    unique_id=unique_id,
                    kind=kind,
                    project_slug=project_slug,
                    state="pending",
                    future=None,
                    event=event,
                    starting_time=datetime.datetime.now(),
                    queue=queue,
                    task=task,
                )
            )

        return unique_id

    def get(self, unique_id: str) -> QueueTaskModel | None:
        """
        Get a process
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            return None
        return element[0]

    def kill(self, unique_id: str) -> None:
        """
        Send a kill process with the event manager
        """
        element = [i for i in self.current if i.unique_id == unique_id]
        if len(element) == 0:
            raise Exception("Process not found")
        element[0].event.set()  # TODO update status to flag the killing
        self.delete(unique_id)  # TODO move this to the cleaning method

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if type(ids) is str:
            ids = [ids]
        for i in [t for t in self.current if t.unique_id in ids]:
            if i.future is None or not i.future.done():
                print("Deleting a unfinished process")
            self.current.remove(i)

    def state(self) -> list[QueueStateTaskModel]:
        """
        Return state of the queue
        """
        return [
            QueueStateTaskModel(
                unique_id=process.unique_id,
                state="done" if process.future and process.future.done() else process.state,
                exception=process.future.exception() if process.future else None,
                kind=process.kind,
            )
            for process in self.current
        ]

    def get_nb_waiting_processes(self, queue: str = "cpu") -> int:
        """
        Number of waiting processes
        """
        return len([f for f in self.current if f.queue == queue and f.state == "pending"])

    def display_info(self, renew: int = 20) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        print(self.state())
        print(
            "waiting",
            self.get_nb_waiting_processes("cpu"),
            self.get_nb_waiting_processes("gpu"),
        )
        return None

    def clean_old_processes(self, timeout: int = 2) -> None:
        """
        Remove old processes
        """
        n = len(self.current)
        self.current = [
            i
            for i in self.current
            if (datetime.datetime.now() - i.starting_time).total_seconds() / 3600 < timeout
        ]
        if n != len(self.current):
            print(f"Cleaned {n - len(self.current)} processes")
        return None

    def restart(self) -> None:
        """
        Restart the queue by getting the executor and closing it
        """
        executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=1000, reuse=True)
        executor.shutdown(wait=False)
        self.manager.shutdown()
        self.manager = Manager()
        self.current = []
