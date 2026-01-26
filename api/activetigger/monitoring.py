from datetime import datetime

import pandas as pd

from activetigger.datamodels import (
    MonitoringLanguageModelsModel,
    MonitoringMetricsModel,
    MonitoringQuickModelsModel,
)
from activetigger.db.manager import DatabaseManager


class Monitoring:
    """
    Manage messages on the interface
    - user messages
    - mail messages
    """

    db_manager: DatabaseManager
    project_slug: str | None

    def __init__(self, db_manager: DatabaseManager, project_slug: str | None = None) -> None:
        self.db_manager = db_manager
        self.project_slug = project_slug

    def register_process(
        self, process_name: str, kind: str, parameters: dict, user_name: str
    ) -> None:
        """
        Start a new monitored process
        """
        events = {"start": datetime.now().isoformat()}
        self.db_manager.monitoring_service.add_process(
            process_name=process_name,
            kind=kind,
            parameters=parameters,
            events=events,
            project_slug=self.project_slug,
            user_name=user_name,
        )

    def close_process(
        self,
        process_name: str,
    ) -> None:
        """
        Close a monitored process
        """
        start_entry = self.db_manager.monitoring_service.get_element_by_process(process_name)
        if start_entry is None:
            raise ValueError(f"Process {process_name} not found")
        events = start_entry.events
        events["end"] = datetime.now().isoformat()

        duration = (datetime.now() - start_entry.time).total_seconds()

        self.db_manager.monitoring_service.update_process(
            process_name=process_name,
            events=events,
            duration=duration,
        )
        print(f"Process {process_name} closed in {duration} seconds")

    def get_completed_processes(
        self, kind: str, username: str | None, limit: int = 100
    ) -> pd.DataFrame:
        """
        Get completed processes of a given kind and user
        """
        processes = self.db_manager.monitoring_service.get_completed_processes(
            kind=kind, username=username, limit=limit
        )
        data = [
            {
                "process_name": process.process_name,
                "kind": process.kind,
                "time": process.time,
                "parameters": process.parameters,
                "events": process.events,
                "project_slug": process.project_slug,
                "user_name": process.user_name,
                "duration": process.duration,
                "username": process.user_name,
            }
            for process in processes
        ]
        df = pd.DataFrame(data)
        return df

    def get_failed_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_running_processes(self):
        raise NotImplementedError("Not implemented yet")

    def get_data(self, kind: str, state: str = "completed", limit: int = 100) -> pd.DataFrame:
        """
        Get monitoring data
        """
        if state == "completed":
            df_processes = self.get_completed_processes(kind=kind, username=None, limit=limit)
        else:
            raise NotImplementedError("Not implemented yet")
        return df_processes

    def get_metrics(self) -> MonitoringMetricsModel:
        """
        Get monitoring metrics
        """
        df_quickmodels = self.get_completed_processes(
            kind="train_quickmodel", username=None, limit=100
        )
        m_quickmodels = MonitoringQuickModelsModel(
            n=len(df_quickmodels),
            mean=0 if len(df_quickmodels) == 0 else df_quickmodels["duration"].mean(),
            std=0 if len(df_quickmodels) < 2 else df_quickmodels["duration"].std(),
        )
        df_languagemodels = self.get_completed_processes(
            kind="train_languagemodel", username=None, limit=100
        )
        m_languagemodels = MonitoringLanguageModelsModel(
            n=len(df_languagemodels),
            mean=0 if len(df_languagemodels) == 0 else df_languagemodels["duration"].mean(),
            std=0 if len(df_languagemodels) < 2 else df_languagemodels["duration"].std(),
        )
        return MonitoringMetricsModel(quickmodels=m_quickmodels, languagemodels=m_languagemodels)
