import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from activetigger.db.models import (
    Logs,
)


class LogsService:
    """
    Database service for logs
    """

    SessionMaker: sessionmaker[Session]

    def __init__(self, sessionmaker: sessionmaker[Session]):
        self.SessionMaker = sessionmaker

    def add_log(self, user: str, action: str, project_slug: str, connect: str):
        session = self.SessionMaker()
        log = Logs(
            user_id=user,
            project_id=project_slug,
            action=action,
            connect=connect,
            time=datetime.datetime.now(),
        )
        session.add(log)
        session.commit()
        session.close()

    def get_logs(self, username: str, project_slug: str, limit: int):
        """
        TODO : secure the log through the project_slug auth
        """
        with self.SessionMaker() as session:
            stmt = select(Logs).order_by(Logs.time.desc()).limit(limit)
            if project_slug != "all":
                stmt = stmt.filter_by(project_id=project_slug)
            if username != "all":
                stmt = stmt.filter_by(user_id=username)

            logs = session.scalars(stmt).all()

        return [
            {
                "id": log.id,
                "time": log.time.strftime("%Y-%m-%d %H:%M:%S"),
                "user": log.user_id,
                "project": log.project_id,
                "action": log.action,
                "connect": log.connect,
            }
            for log in logs
        ]

    def get_last_activity_project(self, project_slug: str):
        with self.SessionMaker() as session:
            stmt = select(Logs).order_by(Logs.time.desc()).limit(1)
            if project_slug != "all":
                stmt = stmt.filter_by(project_id=project_slug)
            logs = session.scalars(stmt).all()

        if len(logs) == 0:
            return None

        return logs[0].time.strftime("%Y-%m-%d %H:%M:%S")

    def get_last_activity_user(self, username: str):
        with self.SessionMaker() as session:
            stmt = select(Logs).order_by(Logs.time.desc()).limit(1)
            if username != "all":
                stmt = stmt.filter_by(user_id=username)
            logs = session.scalars(stmt).all()

        if len(logs) == 0:
            return None

        return logs[0].time.strftime("%Y-%m-%d %H:%M:%S")
