from sqlalchemy import delete, select, update
from sqlalchemy.orm import Session, sessionmaker

from activetigger.db import DBException
from activetigger.db.models import Users


class UsersService:
    SessionMaker: sessionmaker[Session]

    def __init__(self, sessionmaker: sessionmaker[Session]):
        self.SessionMaker = sessionmaker

    def get_user(self, username: str) -> Users:
        # Context manager (with) will automatically close the session outside of its scope
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
        # with .begin(), it also commit the transaction
        with self.SessionMaker.begin() as session:
            user = Users(
                user=username,
                key=password,
                description=role,
                created_by=created_by,
                # time=datetime.datetime.now(), ← This is default value
                contact=contact,
            )
            session.add(user)

    def get_users_created_by(self, username: str):
        """
        get users created by *username*
        """
        with self.SessionMaker() as session:
            stmt = select(Users.user, Users.contact)
            if username != "all":
                stmt = stmt.filter_by(created_by=username)
            stmt = stmt.distinct()
            return {
                row.user: {"contact": row.contact}
                for row in session.scalars(stmt).all()
            }

    def delete_user(self, username: str) -> None:
        with self.SessionMaker.begin() as session:
            _ = session.execute(delete(Users).filter_by(user=username))

    def change_password(self, username: str, password: str) -> None:
        with self.SessionMaker.begin() as session:
            _ = session.execute(
                update(Users).filter_by(user=username).values(key=password)
            )
