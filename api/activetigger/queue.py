import asyncio
import datetime
import logging
import multiprocessing
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any

# manage the executor
from loky import get_reusable_executor  # type: ignore[import]

logger = logging.getLogger("server")
multiprocessing.set_start_method("spawn", force=True)


class Queue:
    """
    Managining parallel processes for computation

    Use Loky as a backend for one common executor

    With 2 waiting queues to differentiate between CPU and GPU jobs
    (to limit concurrency in GPU memory usage)
    """

    path: Path
    max_processes: int = 15
    nb_workers: int
    nb_workers_cpu: int
    nb_workers_gpu: int
    manager: SyncManager
    current: list
    last_restart: datetime.datetime

    def __init__(
        self, nb_workers_cpu: int = 3, nb_workers_gpu: int = 1, path: Path = Path(".")
    ) -> None:
        """
        Initiating the queue
        """
        self.path = path
        self.nb_workers_cpu = nb_workers_cpu
        self.nb_workers_gpu = nb_workers_gpu
        self.nb_workers = nb_workers_cpu + nb_workers_gpu
        self.manager = Manager()
        self.current = []

        # launch a regular update on the queue
        self.task = asyncio.create_task(self._update_queue(timeout=1))
        logger.info("Init Queue")

    def __del__(self) -> None:
        """
        Destructor to close the queue
        """
        if hasattr(self, "manager"):
            self.manager.shutdown()
            logger.info("Queue closed on destructor")
        if hasattr(self, "task"):
            self.task.cancel()
            logger.info("Queue task cancelled on destructor")

    async def _update_queue(self, timeout: int = 1) -> None:
        """
        Update the queue with new tasks every X seconds
        """
        while True:
            # jobs in the executor
            nb_active_processes_gpu = len(
                [i for i in self.current if i["queue"] == "gpu" and i["state"] == "running"]
            )
            nb_active_processes_cpu = len(
                [i for i in self.current if i["queue"] == "cpu" and i["state"] == "running"]
            )

            task_gpu = [i for i in self.current if i["queue"] == "gpu" and i["state"] == "pending"]
            task_cpu = [i for i in self.current if i["queue"] == "cpu" and i["state"] == "pending"]

            # a worker available and possible to have gpu
            if (
                nb_active_processes_gpu < self.nb_workers_gpu
                and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
                and len(task_gpu) > 0
            ):
                print("Add gpu task to the workers")

                executor = get_reusable_executor(
                    max_workers=(self.nb_workers), timeout=1000, reuse=True
                )
                task_gpu[0]["future"] = executor.submit(task_gpu[0]["task"])
                task_gpu[0]["state"] = "running"
                task_gpu[0]["task"] = None
                print("Task added to the workers")

            # a worker available and possible to have cpu
            if (
                nb_active_processes_cpu < self.nb_workers_cpu
                and (nb_active_processes_gpu + nb_active_processes_cpu) < self.nb_workers
                and len(task_cpu) > 0
            ):
                print("Add cpu task to the workers")

                executor = get_reusable_executor(
                    max_workers=(self.nb_workers), timeout=1000, reuse=True
                )
                task_cpu[0]["future"] = executor.submit(task_cpu[0]["task"])
                task_cpu[0]["state"] = "running"
                task_cpu[0]["task"] = None
                print("Task added to the workers")

            print(
                "nb_active_processes_cpu",
                nb_active_processes_cpu,
                "nb_active_processes_gpu",
                nb_active_processes_gpu,
            )

            await asyncio.sleep(timeout)  # Non-blocking sleep

    def add_task(self, kind: str, project_slug: str, task: Any, queue: str = "cpu") -> str:
        """
        Add a task in the queue, first as pending
        """
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
            {
                "unique_id": unique_id,
                "kind": kind,
                "project_slug": project_slug,
                "state": "pending",
                "future": None,
                "event": event,
                "starting_time": datetime.datetime.now(),
                "queue": queue,
                "task": task,
            }
        )

        return unique_id

    def get(self, unique_id: str) -> dict | None:
        """
        Get a process
        """
        element = [i for i in self.current if i["unique_id"] == unique_id]
        if len(element) == 0:
            return None
        return element[0]

    def kill(self, unique_id: str) -> None:
        """
        Send a kill process with the event manager
        """
        element = [i for i in self.current if i["unique_id"] == unique_id]
        if len(element) == 0:
            raise Exception("Process not found")
        element[0]["event"].set()  # TODO update status to flag the killing
        self.delete(unique_id)  # TODO move this to the cleaning method

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if type(ids) is str:
            ids = [ids]
        for i in [t for t in self.current if t["unique_id"] in ids]:
            if i["future"] is None or not i["future"].done():
                print("Deleting a unfinished process")
            self.current.remove(i)

    def state(self) -> dict:
        """
        Return state of the queue
        """
        r = {}
        for process in self.current:
            if process["state"] == "pending":
                info = "pending"
                exception = None
            if process["state"] == "running":
                info = "running"
                exception = None
            if process["future"] is not None and process["future"].done():
                info = "done"
                exception = process["future"].exception()

            r[process["unique_id"]] = {
                "state": info,
                "exception": exception,
                "kind": process["kind"],
            }
        return r

    def get_nb_waiting_processes(self, queue: str = "cpu") -> int:
        """
        Number of waiting processes
        """
        return len([f for f in self.current if f["queue"] == queue and f["state"] == "pending"])

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
            if (datetime.datetime.now() - i["starting_time"]).total_seconds() / 3600 < timeout
        ]
        if n != len(self.current):
            print(f"Cleaned {n - len(self.current)} processes")
        return None

    def restart(self) -> None:
        """
        Restart the queue
        """
        executor = get_reusable_executor(max_workers=(self.nb_workers), timeout=1000, reuse=True)
        executor.shutdown(wait=False)
        self.manager.shutdown()
        self.manager = Manager()
        self.current = []
