from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    TIMESTAMP,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path
from activetigger.functions import get_root_pwd, get_hash
import datetime
import json

Base = declarative_base()


class Project(Base):
    __tablename__ = "projects"
    project_slug = Column(String, primary_key=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    parameters = Column(Text)
    time_modified = Column(TIMESTAMP)
    user = Column(String)


class Scheme(Base):
    __tablename__ = "schemes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    time_modified = Column(
        TIMESTAMP,
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user = Column(String)
    project = Column(String)
    name = Column(String)
    params = Column(Text)


class Annotation(Base):
    __tablename__ = "annotations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    action = Column(String)
    user = Column(String)
    project = Column(String)
    element_id = Column(String)
    scheme = Column(String)
    tag = Column(String)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    key = Column(Text)
    description = Column(Text)
    created_by = Column(String)


class Auth(Base):
    __tablename__ = "auth"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user = Column(String)
    project = Column(String)
    status = Column(String)
    created_by = Column(String)


class Log(Base):
    __tablename__ = "logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    project = Column(String)
    action = Column(String)
    connect = Column(String)


class Token(Base):
    __tablename__ = "tokens"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time_created = Column(TIMESTAMP, server_default=func.current_timestamp())
    token = Column(Text)
    status = Column(String)
    time_revoked = Column(TIMESTAMP)


class Generation(Base):
    __tablename__ = "generations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    time = Column(TIMESTAMP, server_default=func.current_timestamp())
    user = Column(String)
    project = Column(String)
    element_id = Column(String)
    endpoint = Column(String)
    prompt = Column(Text)
    answer = Column(Text)


class DatabaseManager:
    """
    Database management with SQLAlchemy
    """

    def __init__(self, path_db: str):
        self.db_url = f"sqlite:///{path_db}"

        # test if the db exists, else create it
        if not Path(path_db).exists():
            self.create_db()

        # connect the session
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        self.default_user = "server"

        # check if there is a root user, add it
        session = self.Session()
        if not session.query(User).filter_by(user="root").first():
            self.create_root_session()
        session.close()

    def create_db(self):
        print("Create database")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)

    def create_root_session(self) -> None:
        """
        Create root session
        :return: None
        """
        pwd: str = get_root_pwd()
        hash_pwd: bytes = get_hash(pwd)
        self.add_user("root", hash_pwd, "root", "system")

    def add_user(self, user: str, key: str, description: str, created_by: str):
        session = self.Session()
        user = User(user=user, key=key, description=description, created_by=created_by)
        session.add(user)
        session.commit()
        session.close()

    def add_log(self, user: str, action: str, project_slug: str, connect: str):
        session = self.Session()
        log = Log(user=user, project=project_slug, action=action, connect=connect)
        session.add(log)
        session.commit()
        session.close()

    def get_logs(self, username: str, project_slug: str, limit: int):
        session = self.Session()
        if project_slug == "all":
            logs = (
                session.query(Log)
                .filter_by(user=username)
                .order_by(Log.time.desc())
                .limit(limit)
                .all()
            )
        else:
            logs = (
                session.query(Log)
                .filter_by(user=username, project=project_slug)
                .order_by(Log.time.desc())
                .limit(limit)
                .all()
            )
        session.close()
        return [
            {
                "id": log.id,
                "time": log.time,
                "user": log.user,
                "project": log.project,
                "action": log.action,
                "connect": log.connect,
            }
            for log in logs
        ]

    def get_project(self, project_slug: str):
        session = self.Session()
        project = session.query(Project).filter_by(project_slug=project_slug).first()
        session.close()
        if project:
            return project.__dict__
        else:
            return None

    def add_project(self, project_slug: str, parameters: dict, username: str):
        session = self.Session()
        project = Project(
            project_slug=project_slug,
            parameters=json.dumps(parameters),
            time_modified=datetime.datetime.now(),
            user=username,
        )
        session.add(project)
        session.commit()
        session.close()

    def update_project(self, project_slug: str, parameters: dict):
        session = self.Session()
        project = session.query(Project).filter_by(project_slug=project_slug).first()
        project.time_modified = datetime.datetime.now()
        project.parameters = json.dumps(parameters)
        session.commit()
        session.close()

    def existing_projects(self) -> list:
        session = self.Session()
        projects = session.query(Project).all()
        session.close()
        return [project.project_slug for project in projects]

    def add_token(self, token: str, status: str):
        session = self.Session()
        token = Token(token=token, status=status)
        session.add(token)
        session.commit()
        session.close()

    def get_token_status(self, token: str):
        session = self.Session()
        token = session.query(Token).filter_by(token=token).first()
        session.close()
        if token:
            return token.status
        else:
            return None

    def revoke_token(self, token: str):
        session = self.Session()
        token = session.query(Token).filter_by(token=token).first()
        token.time_revoked = datetime.datetime.now()
        token.status = "revoked"
        session.commit()
        session.close()

    def add_scheme(self, project_slug: str, name: str, params: dict, username: str):
        session = self.Session()
        scheme = Scheme(project=project_slug, name=name, params=params, user=username)
        session.add(scheme)
        session.commit()
        session.close()

    def update_scheme(self, project_slug: str, name: str, params: str):
        session = self.Session()
        scheme = (
            session.query(Scheme).filter_by(project=project_slug, name=name).first()
        )
        scheme.params = params
        scheme.time_modified = datetime.datetime.now()
        session.commit()
        session.close()

    def add_annotation(
        self,
        action: str,
        user: str,
        project_slug: str,
        element_id: str,
        scheme: str,
        tag: str,
    ):
        session = self.Session()
        annotation = Annotation(
            action=action,
            user=user,
            project=project_slug,
            element_id=element_id,
            scheme=scheme,
            tag=tag,
        )
        session.add(annotation)
        session.commit()
        session.close()

    def delete_project(self, project_slug: str):
        session = self.Session()
        session.query(Project).filter(Project.project_slug == project_slug).delete()
        session.query(Scheme).filter(Scheme.project == project_slug).delete()
        session.query(Annotation).filter(Annotation.project == project_slug).delete()
        session.query(Auth).filter(Auth.project == project_slug).delete()
        session.query(Generation).filter(Generation.project == project_slug).delete()
        session.query(Log).filter(Log.project == project_slug).delete()
        session.commit()
        session.close()

    def add_generated(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
        prompt: str,
        answer: str,
    ):
        session = self.Session()
        generation = Generation(
            user=user,
            project=project_slug,
            element_id=element_id,
            endpoint=endpoint,
            prompt=prompt,
            answer=answer,
        )
        session.add(generation)
        session.commit()
        session.close()

    def get_generated(self, project_slug: str, n_elements: int):
        session = self.Session()
        generated = (
            session.query(Generation)
            .filter(Generation.project == project_slug)
            .order_by(Generation.time.desc())
            .limit(n_elements)
            .all()
        )
        session.close()
        return [[el.element_id, el.prompt, el.answer, el.endpoint] for el in generated]

    def get_distinct_users(self, project_slug: str, timespan: int | None):
        session = self.Session()
        if timespan:
            time_threshold = datetime.datetime.now() - datetime.timedelta(
                seconds=timespan
            )
            users = (
                session.query(Generation.user)
                .filter(
                    Generation.project == project_slug, Generation.time > time_threshold
                )
                .distinct()
                .all()
            )

        else:
            users = (
                session.query(Generation.user)
                .filter(Generation.project == project_slug)
                .distinct()
                .all()
            )
        session.close()
        return [u.user for u in users]

    def get_project_auth(self, project_slug: str):
        session = self.Session()
        auth = session.query(Auth).filter(Auth.project == project_slug).all()
        session.close()
        return {el.user: el.status for el in auth}

    def add_auth(self, project_slug: str, user: str, status: str):
        session = self.Session()
        auth = (
            session.query(Auth)
            .filter(Auth.project == project_slug, Auth.user == user)
            .first()
        )
        if auth:
            auth.status = status
        else:
            auth = Auth(project=project_slug, user=user, status=status)
            session.add(auth)
        session.commit()
        session.close()

    def delete_auth(self, project_slug: str, user: str):
        session = self.Session()
        session.query(Auth).filter(
            Auth.project == project_slug, Auth.user == user
        ).delete()
        session.commit()
        session.close()

    def get_user_projects(self, username: str):
        session = self.Session()
        result = (
            session.query(
                Auth.project,
                Auth.status,
                Project.parameters,
                Project.user,
                Project.time_created,
            )
            .join(Project, Auth.project == Project.project_slug)
            .filter(Auth.user == username)
            .all()
        )
        session.close()
        return [row for row in result]

    def get_user_auth(self, username: str, project_slug: str = None):
        session = self.Session()
        if project_slug is None:
            result = (
                session.query(Auth.user, Auth.status)
                .filter(Auth.user == username)
                .all()
            )
        else:
            result = (
                session.query(Auth.user, Auth.status)
                .filter(Auth.user == username, Auth.project == project_slug)
                .all()
            )
        session.close()
        return [[row[0], row[1]] for row in result]

    def get_users(self):
        session = self.Session()
        result = session.query(User.user).distinct().all()
        session.close()
        return [row.user for row in result]

    def add_user(self, username: str, password: str, role: str, created_by: str):
        session = self.Session()
        user = User(
            user=username, key=password, description=role, created_by=created_by
        )
        session.add(user)
        session.commit()
        session.close()

    def delete_user(self, username: str):
        session = self.Session()
        session.query(User).filter(User.user == username).delete()
        session.commit()
        session.close()

    def get_user(self, username: str):
        session = self.Session()
        user = session.query(User).filter(User.user == username).first()
        session.close()
        return {"key": user.key, "description": user.description}

    def get_scheme_elements(self, project_slug: str, scheme: str, actions: list[str]):
        """
        Get last annotation for each element id for a project/scheme
        """
        session = self.Session()
        query = (
            session.query(
                Annotation.element_id,
                Annotation.tag,
                Annotation.user,
                Annotation.time,
                func.max(Annotation.time),
            )
            .filter(
                Annotation.scheme == scheme,
                Annotation.project == project_slug,
                Annotation.action.in_(actions),
            )
            .group_by(Annotation.element_id)
            .order_by(func.max(Annotation.time).desc())
        )

        # Execute the query and fetch all results
        results = query.all()
        session.close()
        return [[row.element_id, row.tag, row.user, row.time] for row in results]

    def get_coding_users(self, scheme: str, project_slug: str):
        session = self.Session()
        distinct_users = (
            session.query(Annotation.user)
            .filter(Annotation.project == project_slug, Annotation.scheme == scheme)
            .distinct()
            .all()
        )
        session.close()
        return [u for u in distinct_users]

    def get_recent_annotations(
        self, project_slug: str, user: str, scheme: str, limit: int
    ):
        session = self.Session()
        if user == "all":
            recent_annotations = (
                session.query(Annotation.element_id)
                .filter(
                    Annotation.project == project_slug,
                    Annotation.scheme == scheme,
                    Annotation.action == "add",
                )
                .order_by(Annotation.time.desc())
                .limit(limit)
                .distinct()
                .all()
            )

        else:
            recent_annotations = (
                session.query(Annotation.element_id)
                .filter(
                    Annotation.project == project_slug,
                    Annotation.scheme == scheme,
                    Annotation.user == user,
                    Annotation.action == "add",
                )
                .order_by(Annotation.time.desc())
                .limit(limit)
                .distinct()
                .all()
            )
        return [u for u in recent_annotations]

    def get_annotations_by_element(
        self, project_slug: str, scheme: str, element_id: str, limit: int = 10
    ):
        session = self.Session()
        annotations = (
            session.query(
                Annotation.tag, Annotation.action, Annotation.user, Annotation.time
            )
            .filter(
                Annotation.project == project_slug,
                Annotation.scheme == scheme,
                Annotation.element_id == element_id,
            )
            .order_by(Annotation.time.desc())
            .limit(limit)
            .all()
        )
        return [[a.tag, a.action, a.user, a.time] for a in annotations]

    def post_annotation(
        self,
        project_slug: str,
        scheme: str,
        element_id: str,
        tag: str,
        user: str,
        action: str,
    ):
        session = self.Session()
        annotation = Annotation(
            action=action,
            user=user,
            project=project_slug,
            element_id=element_id,
            scheme=scheme,
            tag=tag,
        )
        session.add(annotation)
        session.commit()
        session.close()

    def available_schemes(self, project_slug: str):
        session = self.Session()
        schemes = (
            session.query(Scheme.name, Scheme.params)
            .filter(Scheme.project == project_slug)
            .distinct()
            .all()
        )
        session.close()
        return [[s.name, s.params] for s in schemes]

    def delete_scheme(self, project_slug: str, name: str):
        session = self.Session()
        session.query(Scheme).filter(
            Scheme.name == name, Scheme.project == project_slug
        ).delete()
        session.commit()
        session.close()

    def get_table_annotations_users(self, project_slug: str, scheme: str):
        session = self.Session()
        subquery = (
            select(
                Annotation.id,
                Annotation.user,
                func.max(Annotation.time).label("last_timestamp"),
            )
            .where(Annotation.project == project_slug, Annotation.scheme == scheme)
            .group_by(Annotation.element_id, Annotation.user)
            .subquery()
        )
        query = select(
            Annotation.element_id, Annotation.tag, Annotation.user, Annotation.time
        ).join(subquery, Annotation.id == subquery.c.id)

        results = session.execute(query).fetchall()
        session.close()
        return [[row.element_id, row.tag, row.user, row.time] for row in results]
