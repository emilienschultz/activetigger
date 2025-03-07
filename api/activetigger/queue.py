import concurrent.futures
import datetime
import logging
import multiprocessing
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Any, Callable

import psutil

logger = logging.getLogger("server")


class Queue:
    """
    Managining parallel processes for computation
    Jobs with  concurrent.futures.ProcessPoolExecutor

    TODO : better management of failed processes
    """

    nb_workers: int
    executor: concurrent.futures.ProcessPoolExecutor
    manager: SyncManager
    current: dict
    path: Path
    last_restart: datetime.datetime

    def __init__(self, nb_workers: int = 4, path: Path = Path(".")) -> None:
        """
        Initiating the queue
        """
        self.nb_workers = nb_workers
        self.path = path
        self.last_restart = datetime.datetime.now()
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.nb_workers, mp_context=multiprocessing.get_context("spawn")
        )  # manage parallel processes
        self.manager = Manager()  # communicate within processes
        self.current = {}  # keep track of the current stack

        logger.info("Init Queue")

    def close(self) -> None:
        """
        Close the executor
        """
        # TODO : tuer les workers
        self.executor.shutdown(cancel_futures=True, wait=False)
        self.manager.shutdown()
        logger.info("Close queue")
        print("Queue closes")

    def restart(self) -> None:
        """
        Restart the executor
        """
        print("Restarting executor")
        self.executor.shutdown(cancel_futures=True)
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.nb_workers, mp_context=multiprocessing.get_context("spawn")
        )
        self.last_restart = datetime.datetime.now()

    def check(self, renew: int = 20) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        try:
            # restart the executor after X min to clear
            if (datetime.datetime.now() - self.last_restart).seconds > renew * 60:
                # only if no process is running
                if all(self.current[p]["future"].done() for p in self.current):
                    self.restart()
            self.executor.submit(lambda: None)
            print("workers", self.get_workers_info())
        except Exception:
            self.restart()
            logger.error("Restart executor")
            print("Problem with executor ; restart")

    def get_workers_info(self) -> dict:
        """
        Get info on the workers
        """
        process_info = {}
        for pid in self.executor._processes.keys():
            try:
                p = psutil.Process(pid)
                process_info[pid] = {
                    "alive": p.is_running(),
                    "memory_mb": p.memory_info().rss / (1024 * 1024),  # Convertir en Mo
                }
            except psutil.NoSuchProcess:
                process_info[pid] = {"alive": False, "memory_mb": None}
        return process_info

    def add_task(self, kind: str, project_slug: str, task: Any) -> str:
        """
        Add a task to the queue
        """
        # generate a unique id
        unique_id = str(uuid.uuid4())
        event = self.manager.Event()
        task.event = event
        task.unique_id = unique_id
        task.path_process = self.path.joinpath(project_slug)

        # send the process to the executor
        try:
            future = self.executor.submit(task)
        except Exception as e:
            print("Error submitting task: ", e)
            logger.error(f"Error submitting task: {e}")
            return "error"

        # save in the stack
        self.current[unique_id] = {
            "kind": kind,
            "project_slug": project_slug,
            "future": future,
            "event": event,
            "starting_time": datetime.datetime.now(),
        }
        return unique_id

    def add(self, kind: str, project_slug: str, func: Callable, args: dict) -> str:
        """
        Add a function in the queue
        """
        unique_id = str(uuid.uuid4())
        event = self.manager.Event()
        args["event"] = event
        args["unique_id"] = unique_id

        # send the process to the executor
        try:
            future = self.executor.submit(func, **args)
        except Exception as e:
            print("Error submitting task: ", e)
            logger.error(f"Error submitting task: {e}")
            return "error"

        # save in the stack
        self.current[unique_id] = {
            "kind": kind,
            "project_slug": project_slug,
            "future": future,
            "event": event,
            "starting_time": datetime.datetime.now(),
        }
        return unique_id

    def kill(self, unique_id: str) -> None:
        """
        Send a kill process with the event manager
        """
        if unique_id not in self.current:
            raise Exception("Process not found")
        self.current[unique_id]["event"].set()  # TODO update status to flag the killing
        self.delete(unique_id)  # TODOmove this to the cleaning method

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if type(ids) is str:
            ids = [ids]
        for i in ids:
            if not self.current[i]["future"].done():
                print("Deleting a unfinished process")
            del self.current[i]

    def state(self) -> dict:
        """
        Return state of the queue
        """
        r = {}
        for f in self.current:
            if self.current[f]["future"].running():
                info = "running"
                exception = None
            else:
                info = "done"
                exception = self.current[f]["future"].exception()
            r[f] = {
                "state": info,
                "exception": exception,
                "kind": self.current[f]["kind"],
            }
        return r

    def get_nb_active_processes(self) -> int:
        """
        Number of active processes
        """
        return len([f for f in self.current if self.current[f]["future"].running()])
