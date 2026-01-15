import uuid

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from activetigger.config import config
from activetigger.db import DBException
from activetigger.db.generations import GenerationsService
from activetigger.db.languagemodels import ModelsService
from activetigger.db.logs import LogsService
from activetigger.db.messages import MessagesService
from activetigger.db.models import Base
from activetigger.db.monitoring import MonitoringService
from activetigger.db.projects import ProjectsService
from activetigger.db.users import UsersService
from activetigger.functions import get_hash, get_root_pwd


def set_sqlite_pragma(dbapi_connection, _):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class DatabaseManager:
    """
    Database management with SQLAlchemy
    """

    engine: Engine
    SessionMaker: sessionmaker[Session]
    default_user: str
    users_service: UsersService
    projects_service: ProjectsService

    def __init__(self):
        # priority to environ if set
        db_url = config.database_url

        # connect the session
        print(f"connecting to DB ${db_url}")
        self.engine = create_engine(db_url)

        # enable foreign key verification in sqlite
        if db_url.startswith("sqlite"):
            event.listen(self.engine, "connect", set_sqlite_pragma)

        self.SessionMaker = sessionmaker(bind=self.engine)
        self.default_user = "server"
        self.users_service = UsersService(self.SessionMaker)
        self.projects_service = ProjectsService(self.SessionMaker)
        self.generations_service = GenerationsService(self.SessionMaker)
        self.language_models_service = ModelsService(self.SessionMaker)
        self.logs_service = LogsService(self.SessionMaker)
        self.messages_service = MessagesService(self.SessionMaker)
        self.monitoring_service = MonitoringService(self.SessionMaker)

        # Create tables if not already present
        Base.metadata.create_all(self.engine)

        # check if there is a root user, add it
        try:
            _ = self.users_service.get_user("system")
        except DBException:
            self.create_system_session()

        # check if there is a root user, add it
        try:
            _ = self.users_service.get_user("root")
        except DBException:
            self.create_root_session()

        # check if there is a demo user, add it
        try:
            _ = self.users_service.get_user("demo")
        except DBException:
            self.create_demo_session()

    def create_root_session(self) -> None:
        """
        Create root session
        :return: None
        """
        pwd: str = config.root_password if config.root_password is not None else get_root_pwd()
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("root", hash_pwd.decode("utf8"), "root", "system")

    def create_system_session(self) -> None:
        """
        Create root session
        :return: None
        """
        # use a random password
        pwd: str = str(uuid.uuid4())
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("system", hash_pwd.decode("utf8"), "system", "system")

    def create_demo_session(self) -> None:
        """
        Create demo session
        :return: None
        """
        # use a random password
        pwd: str = "demo"
        hash_pwd: bytes = get_hash(pwd)
        self.users_service.add_user("demo", hash_pwd.decode("utf8"), "demo", "system")
