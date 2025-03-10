import asyncio
import concurrent.futures
import datetime
import logging
import multiprocessing
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any

import psutil

# manage the executor
from loky import get_reusable_executor  # type: ignore[import]

logger = logging.getLogger("server")
multiprocessing.set_start_method("spawn", force=True)


class Queue:
    """
    Managining parallel processes for computation

    Use Loky as a backend for the executor

    A waiting queue is needed to differentiate between CPU and GPU jobs
    otherwise there is concurrency in the memory usage
    """

    path: Path
    max_processes: int = 15
    nb_workers: int
    nb_workers_cpu: int
    nb_workers_gpu: int
    executor: concurrent.futures.ProcessPoolExecutor
    executor_gpu: concurrent.futures.ProcessPoolExecutor
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
        self.executor = get_reusable_executor(max_workers=(self.nb_workers))
        self.manager = Manager()
        self.current = []

        # launch the queue with a regular update
        self.task = asyncio.create_task(self._update_queue(timeout=1))
        logger.info("Init Queue")

    async def _update_queue(self, timeout: int = 1) -> None:
        """
        Update the queue with new tasks every X seconds
        """
        while True:
            nb_active_processes_gpu = len(
                [
                    i
                    for i in self.current
                    if i["queue"] == "gpu" and i["state"] == "running"
                ]
            )
            nb_active_processes_cpu = len(
                [
                    i
                    for i in self.current
                    if i["queue"] == "cpu" and i["state"] == "running"
                ]
            )

            task_gpu = [
                i
                for i in self.current
                if i["queue"] == "gpu" and i["state"] == "pending"
            ]
            task_cpu = [
                i
                for i in self.current
                if i["queue"] == "cpu" and i["state"] == "pending"
            ]
            # a worker available and possible to have gpu
            if (
                nb_active_processes_gpu < self.nb_workers_gpu
                and (nb_active_processes_gpu + nb_active_processes_cpu)
                < self.nb_workers
                and len(task_gpu) > 0
            ):
                print("Add gpu task to the workers")

                task_gpu[0]["state"] = "running"
                task_gpu[0]["future"] = await asyncio.to_thread(
                    self.executor.submit, task_gpu[0]["task"]
                )
                task_gpu[0]["task"] = None
                print("element added, continue")

            # a worker available and possible to have cpu
            if (
                nb_active_processes_cpu < self.nb_workers_cpu
                and (nb_active_processes_gpu + nb_active_processes_cpu)
                < self.nb_workers
                and len(task_cpu) > 0
            ):
                print("Add cpu task to the workers")

                task_cpu[0]["future"] = await asyncio.to_thread(
                    self.executor.submit, task_cpu[0]["task"]
                )
                task_cpu[0]["state"] = "running"
                task_cpu[0]["task"] = None

            # self.clean_old_processes(timeout=2)

            print(
                "nb_active_processes_cpu",
                nb_active_processes_cpu,
                "nb_active_processes_gpu",
                nb_active_processes_gpu,
            )

            await asyncio.sleep(timeout)  # Non-blocking sleep

    def clean_old_processes(self, timeout: int = 2) -> None:
        """
        Remove old processes
        """
        n = len(self.current)
        self.current = [
            i
            for i in self.current
            if (datetime.datetime.now() - i["starting_time"]).total_seconds() / 3600
            < timeout
        ]
        if n != len(self.current):
            print(f"Cleaned {n - len(self.current)} processes")
        return None

    def add_task(
        self, kind: str, project_slug: str, task: Any, queue: str = "cpu"
    ) -> str:
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
        return len(
            [f for f in self.current if f["queue"] == queue and f["state"] == "pending"]
        )

    def get_workers_info(self) -> dict:
        """
        Get info on the workers
        """
        process_info = {}

        # extract the info
        for pid in self.executor._processes.keys():
            try:
                p = psutil.Process(pid)
                process_info[pid] = {
                    "alive": p.is_running(),
                    "memory_mb": p.memory_info().rss / (1024 * 1024),
                }
            except psutil.NoSuchProcess:
                process_info[pid] = {"alive": False, "memory_mb": None}

        return process_info

    def display_info(self, renew: int = 20) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        print("workers", self.get_workers_info())
        print(
            "waiting",
            self.get_nb_waiting_processes("cpu"),
            self.get_nb_waiting_processes("gpu"),
        )
        return None

    def close(self) -> None:
        """
        Close the executor
        """
        if self.executor is not None:
            self.executor.shutdown(wait=False)
        self.manager.shutdown()
        logger.info("Close queue")
        print("Queue closes")


# class QueueOld:
#     """
#     Managining parallel processes for computation
#     Jobs with  concurrent.futures.ProcessPoolExecutor

#     Need to differentiate between CPU and GPU jobs
#     otherwise there is concurrency in the memory usage

#     There is 2 different pools managing 2 queue
#     """

#     path: Path
#     max_waiting_processes: int = 15
#     nb_workers_cpu: int
#     nb_workers_gpu: int
#     executor: concurrent.futures.ProcessPoolExecutor | None
#     executor_gpu: concurrent.futures.ProcessPoolExecutor | None
#     manager: SyncManager
#     current: dict
#     last_restart: datetime.datetime

#     def __init__(
#         self, nb_workers_cpu: int = 3, nb_workers_gpu: int = 1, path: Path = Path(".")
#     ) -> None:
#         """
#         Initiating the queue
#         """
#         self.executor = None
#         self.executor_gpu = None
#         self.nb_workers_cpu = nb_workers_cpu
#         self.nb_workers_gpu = nb_workers_gpu
#         self.path = path
#         self.manager = Manager()  # communicate within processes
#         self.current = {}  # keep track of the current stack

#         # start the executor
#         self.start_pools()
#         logger.info("Init Queue")

#     def start_pools(self) -> None:
#         """
#         Create the pools
#         """
#         # close the previous executor
#         if self.executor is not None:
#             self.executor.shutdown(cancel_futures=True)
#         if self.executor_gpu is not None:
#             self.executor_gpu.shutdown(cancel_futures=True)

#         context = multiprocessing.get_context("spawn")
#         self.executor = concurrent.futures.ProcessPoolExecutor(
#             max_workers=self.nb_workers_cpu, mp_context=context
#         )
#         if self.nb_workers_gpu > 0:
#             print("Init GPU pool")
#             self.executor_gpu = concurrent.futures.ProcessPoolExecutor(
#                 max_workers=self.nb_workers_gpu, mp_context=context
#             )
#         self.last_restart = datetime.datetime.now()
#         print("Pool initialized")

#     def close(self) -> None:
#         """
#         Close the executor
#         """
#         if self.executor is not None:
#             self.executor.shutdown(cancel_futures=True, wait=False)
#         if self.executor_gpu is not None:
#             self.executor_gpu.shutdown(cancel_futures=True, wait=False)
#         self.manager.shutdown()
#         logger.info("Close queue")
#         print("Queue closes")

#     def check(self, renew: int = 20) -> None:
#         """
#         Check if the exector still works
#         if not, recreate it
#         """
#         try:
#             # restart the executor after X min to clear
#             if (datetime.datetime.now() - self.last_restart).seconds > renew * 60:
#                 # only if no process is running
#                 if all(self.current[p]["future"].done() for p in self.current):
#                     self.start_pools()
#             # test if general executor ok
#             if self.executor is not None:
#                 self.executor.submit(lambda: None)
#             print("workers", self.get_workers_info("cpu"), self.get_workers_info("gpu"))
#         except Exception:
#             self.start_pools()
#             logger.error("Restart executor")
#             print("Problem with executor ; restart")

#     def get_workers_info(self, queue: str = "cpu") -> dict:
#         """
#         Get info on the workers
#         """
#         process_info = {}

#         # select the queue
#         if queue == "cpu":
#             if self.executor is None:
#                 return {}
#             pids = self.executor._processes.keys()
#         elif queue == "gpu":
#             if self.executor_gpu is None:
#                 return {}
#             pids = self.executor_gpu._processes.keys()

#         # extract the info
#         for pid in pids:
#             try:
#                 p = psutil.Process(pid)
#                 process_info[pid] = {
#                     "alive": p.is_running(),
#                     "memory_mb": p.memory_info().rss / (1024 * 1024),
#                 }
#             except psutil.NoSuchProcess:
#                 process_info[pid] = {"alive": False, "memory_mb": None}

#         return process_info

#     def submit_to_queue(self, queue: str, task: Any) -> concurrent.futures.Future:
#         """
#         Submit a task to the queue
#         """
#         # if not gpu, send to cpu
#         if queue == "gpu" and self.nb_workers_gpu == 0:
#             queue = "cpu"

#         # select the queue
#         if queue == "cpu":
#             if self.executor is None:
#                 raise Exception("Executor CPU not started")
#             future = self.executor.submit(task)
#         elif queue == "gpu":
#             if self.executor_gpu is None:
#                 raise Exception("Executor GPU not started")
#             future = self.executor_gpu.submit(task)
#         else:
#             raise Exception("Queue not found", queue)
#         return future

#     def add_task(
#         self, kind: str, project_slug: str, task: Any, queue: str = "cpu"
#     ) -> str:
#         """
#         Add a task to one of the 2 queue
#         """
#         # test if the queue is not full
#         if self.get_nb_waiting_processes(queue) > self.max_waiting_processes:
#             raise Exception("Queue is full. Wait for process to finish.")

#         # generate a unique id
#         unique_id = str(uuid.uuid4())
#         event = self.manager.Event()
#         task.event = event
#         task.unique_id = unique_id
#         task.path_process = self.path.joinpath(project_slug)

#         # select the executor
#         future = self.submit_to_queue(queue, task)

#         # save in the stack
#         self.current[unique_id] = {
#             "kind": kind,
#             "project_slug": project_slug,
#             "future": future,
#             "event": event,
#             "starting_time": datetime.datetime.now(),
#             "queue": queue,
#         }

#         return unique_id

#     def kill(self, unique_id: str) -> None:
#         """
#         Send a kill process with the event manager
#         """
#         if unique_id not in self.current:
#             raise Exception("Process not found")
#         self.current[unique_id]["event"].set()  # TODO update status to flag the killing
#         self.delete(unique_id)  # TODO move this to the cleaning method

#     def delete(self, ids: str | list) -> None:
#         """
#         Delete completed elements from the stack
#         """
#         if type(ids) is str:
#             ids = [ids]
#         for i in ids:
#             if not self.current[i]["future"].done():
#                 print("Deleting a unfinished process")
#             del self.current[i]

#     def state(self) -> dict:
#         """
#         Return state of the queue
#         """
#         r = {}
#         for f in self.current:
#             if self.current[f]["future"].running():
#                 info = "running"
#                 exception = None
#             else:
#                 info = "done"
#                 exception = self.current[f]["future"].exception()
#             r[f] = {
#                 "state": info,
#                 "exception": exception,
#                 "kind": self.current[f]["kind"],
#             }
#         return r

#     def get_nb_active_processes(self) -> int:
#         """
#         Number of active processes
#         """
#         return len([f for f in self.current if self.current[f]["future"].running()])

#     def get_nb_waiting_processes(self, queue: str = "cpu") -> int:
#         """
#         Number of waiting processes
#         """
#         return len(
#             [
#                 f
#                 for f in self.current
#                 if self.current[f]["queue"] == queue
#                 and not self.current[f]["future"].running()
#                 and not self.current[f]["future"].done()
#             ]
#         )
