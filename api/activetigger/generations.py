import pandas as pd
from pandas import DataFrame

from activetigger.db import DatabaseManager
from activetigger.queue import Queue


class Generations:
    """
    Class to manage generation data
    """

    queue: Queue
    generating: dict
    db_manager: DatabaseManager

    def __init__(self, queue: Queue, db_manager: DatabaseManager) -> None:
        self.queue = queue
        self.db_manager = db_manager
        self.generating = {}  # user:{"unique_id", "number","api"}

    def update_generations(self):
        """
        Manage the process launched by the class
        Only one possible by user
        """
        for name in self.generating.copy():
            unique_id = self.generating[name]["unique_id"]

            # case the process have been canceled, clean
            if unique_id not in self.queue.current:
                del self.generating[name]
                continue

            # else check its state
            if self.queue.current[unique_id]["future"].done():
                r = self.queue.current[unique_id]["future"].result()
                if "error" in r:
                    print("Error in the generating process", unique_id)
                else:
                    results = r["success"]
                    for row in results:
                        self.add(
                            user=row["user"],
                            project_slug=row["project_slug"],
                            element_id=row["element_id"],
                            endpoint=row["endpoint"],
                            prompt=row["prompt"],
                            answer=row["answer"],
                        )
                    self.queue.delete(unique_id)
                    del self.generating[name]

    def add(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
        prompt: str,
        answer: str,
    ) -> None:
        """
        Add a generated element in the database
        """
        self.db_manager.add_generated(
            user=user,
            project_slug=project_slug,
            element_id=element_id,
            endpoint=endpoint,
            prompt=prompt,
            answer=answer,
        )
        return None

    def get_generated(
        self,
        project_slug: str,
        username: str,
        n_elements: str,
    ) -> DataFrame:
        """
        Get generated elements from the database
        """
        result = self.db_manager.get_generated(
            project_slug=project_slug, username=username, n_elements=n_elements
        )
        df = pd.DataFrame(
            result, columns=["time", "index", "prompt", "answer", "endpoint"]
        )
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        return df
