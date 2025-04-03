import datetime
from collections.abc import Sequence

from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from activetigger.db import DBException
from activetigger.db.models import (
    Annotations,
    Logs,
    Projects,
    Users,
)


class UsersService:
    SessionMaker: sessionmaker[Session]

    def __init__(self, sessionmaker: sessionmaker[Session]):
        self.SessionMaker = sessionmaker

    def get_user(self, username: str) -> Users:
        with self.SessionMaker() as session:
            user = session.scalars(select(Users).filter_by(user=username)).first()
            if user is None:
                raise DBException(f"User {username} not found")
            return user

    def add_user(
        self,
        username: str,
        password: str,
        role: str,
        created_by: str,
        contact: str = "",
    ) -> None:
        with self.SessionMaker.begin() as session:
            user = Users(
                user=username,
                key=password,
                description=role,
                created_by=created_by,
                contact=contact,
                deactivated=None,
            )
            session.add(user)

    def get_users_created_by(
        self, username: str, active: bool = True
    ) -> dict[str, dict]:
        """
        get users created by *username*
        """
        with self.SessionMaker() as session:
            stmt = select(Users.user, Users.contact)
            if username != "all":
                stmt = stmt.filter_by(created_by=username)
            if active:
                stmt = stmt.filter(Users.deactivated.is_(None))
            stmt = stmt.distinct()
            return {row[0]: {"contact": row[1]} for row in session.execute(stmt).all()}

    def delete_user(self, username: str) -> None:
        """
        Deletion means :
        - deactivate the user since we need to keep his data structure
        - delete his auth
        """
        with self.SessionMaker.begin() as session:
            # deactivate the account
            session.execute(
                update(Users)
                .where(Users.user == username)
                .values(deactivated=datetime.datetime.now())
            )
            # delete his auths
            # session.execute(delete(Auths).filter_by(user_id=username))

    def change_password(self, username: str, password: str) -> None:
        """
        Change password for a specific user
        """
        with self.SessionMaker.begin() as session:
            _ = session.execute(
                update(Users).filter_by(user=username).values(key=password)
            )

    def get_distinct_users(
        self, project_slug: str, timespan: int | None
    ) -> Sequence[Users]:
        with self.SessionMaker() as session:
            stmt = (
                select(Projects.user)
                .join_from(Projects, Users)
                .where(Projects.project_slug == project_slug)
                .distinct()
            )
            if timespan:
                time_threshold = datetime.datetime.now() - datetime.timedelta(
                    seconds=timespan
                )
                stmt = stmt.join(Annotations).where(
                    Annotations.time > time_threshold,
                )
            return session.scalars(stmt).all()

    def get_current_users(self, timespan: int = 600):
        with self.SessionMaker() as session:
            time_threshold = datetime.datetime.now() - datetime.timedelta(
                seconds=timespan
            )
            users = (
                session.query(Logs.user_id)
                .filter(Logs.time > time_threshold)
                .distinct()
                .all()
            )
            return [u.user_id for u in users]

    def get_coding_users(self, scheme: str, project_slug: str) -> Sequence[Users]:
        with self.SessionMaker() as session:
            distinct_users = session.scalars(
                select(Annotations.user)
                .join_from(Annotations, Users)
                .where(
                    Annotations.project_id == project_slug,
                    Annotations.scheme_id == scheme,
                )
                .distinct()
            ).all()
            return distinct_users

    def get_user_created_projects(self, username: str) -> list[str]:
        """
        Projects user created
        """
        with self.SessionMaker() as session:
            result = session.execute(
                select(
                    Projects.project_slug,
                ).where(Projects.user_id == username)
            ).all()
            return [row[0] for row in result]
