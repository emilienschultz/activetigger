import concurrent.futures
import datetime
import logging
import uuid
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from typing import Callable

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

    def __init__(self, nb_workers: int = 4) -> None:
        """
        Initiating the queue
        """
        self.nb_workers = nb_workers
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.nb_workers
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

    # def check_failed_processes(self, future):
    #     """
    #     Check if a future failed.
    #     """
    #     try:
    #         result = future.result()  # Will raise if the task failed
    #         return False
    #     except Exception as e:
    #         return True

    def check(self) -> None:
        """
        Check if the exector still works
        if not, recreate it
        """
        try:
            self.executor.submit(lambda: None)
        except Exception:
            self.executor.shutdown(cancel_futures=True)
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.nb_workers
            )
            logger.error("Restart executor")
            print("Problem with executor ; restart")

    def add_task(self, kind: str, project_slug: str, task: Callable) -> str:
        """
        Temporary
        """
        # generate a unique id
        unique_id = str(uuid.uuid4())
        event = self.manager.Event()
        task.event = event
        task.unique_id = unique_id

        # send the process to the executor
        try:
            future = self.executor.submit(task)
        except Exception as e:
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
        Add new element to queue
        - launch the function func and args as a subprocess
        - push the process in the queue
        """
        # generate a unique id
        unique_id = str(uuid.uuid4())

        # create an event to control the process
        event = self.manager.Event()
        args["event"] = event
        args["unique_id"] = unique_id

        # send the process to the executor
        try:
            future = self.executor.submit(func, **args)
        except Exception as e:
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

    def kill(self, unique_id: str) -> dict:
        """
        Send a kill process with the event manager
        """
        if unique_id not in self.current:
            return {"error": "Id does not exist"}
        self.current[unique_id]["event"].set()  # TODO update status to flag the killing
        self.delete(unique_id)  # TODOmove this to the cleaning method
        return {"success": "Process killed"}

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
