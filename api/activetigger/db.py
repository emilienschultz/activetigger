import datetime
import json
import logging
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, List

from sqlalchemy import (
    Integer,
    DateTime,
    Text,
    create_engine,
    sessionmaker,
    func,
    select,
    update,
    delete,
    Session,
    ForeignKey,
)
from sqlalchemy.types import JSON
from sqlalchemy.exc import IntegrityError, NoResultFound
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase, relationship

from activetigger.datamodels import UserInDBModel
from activetigger.functions import get_hash, get_root_pwd


class DBException(Exception):
    pass


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Projects(Base):
    __tablename__ = "projects"

    project_slug: Mapped[str] = mapped_column(primary_key=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    parameters: Mapped[dict[str, Any]]
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped["Users"] = relationship("Users", cascade="all, delete-orphan")
    schemes: Mapped[List["Schemes"]] = relationship(
        "Schemes", cascade="all, delete-orphan", back_populates="project"
    )
    annotations: Mapped[List["Annotations"]] = relationship(
        "Annotations", cascade="all, delete-orphan", back_populates="project"
    )
    auths: Mapped[List["Auths"]] = relationship(
        "Auths", cascade="all, delete-orphan", back_populates="project"
    )
    logs: Mapped[List["Logs"]] = relationship(
        "Logs", cascade="all, delete-orphan", back_populates="project"
    )
    generations: Mapped[List["Generations"]] = relationship(
        "Generations", cascade="all, delete-orphan", back_populates="project"
    )
    features: Mapped[List["Features"]] = relationship(
        "Features", cascade="all, delete-orphan", back_populates="project"
    )
    models: Mapped[List["Models"]] = relationship(
        "Models", cascade="all, delete-orphan", back_populates="project"
    )


class Users(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[str]
    key: Mapped[str]
    description: Mapped[str]
    contact: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str]
    projects: Mapped[List[Projects]] = relationship(back_populates="user")


class Schemes(Base):
    __tablename__ = "schemes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="scheme")
    models: Mapped[List["Models"]] = relationship()
    name: Mapped[str]
    params: Mapped[dict[str, Any]]


class Annotations(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    dataset: Mapped[str]
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="annotations")
    element_id: Mapped[str]
    scheme: Mapped[Schemes] = relationship()
    annotation: Mapped[str]
    comment: Mapped[str] = mapped_column(Text)


class Auths(Base):
    __tablename__ = "auth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="auths")
    status: Mapped[str]
    created_by: Mapped[str]


class Logs(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="logs")
    action: Mapped[str]
    connect: Mapped[str]


class Tokens(Base):
    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    token: Mapped[str]
    status: Mapped[str]
    time_revoked: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))


class Generations(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="generations")
    element_id: Mapped[str]
    endpoint: Mapped[str]
    prompt: Mapped[str]
    answer: Mapped[str]


class Features(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="features")
    name: Mapped[str]
    kind: Mapped[str]
    parameters: Mapped[str]
    data: Mapped[str]


class Models(Base):
    __tablename__ = "models"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    time_modified: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.current_timestamp(),
        onupdate=func.current_timestamp(),
    )
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.id"))
    project: Mapped[Projects] = relationship(back_populates="models")
    scheme_id: Mapped[int] = mapped_column(ForeignKey("schemes.id"))
    scheme: Mapped[Schemes] = relationship(back_populates="models")
    kind: Mapped[str]
    name: Mapped[str]
    parameters: Mapped[str]
    path: Mapped[str]
    status: Mapped[str]
    statistics: Mapped[str]
    test: Mapped[str]


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
        self.Session = sessionmaker(self.engine)
        self.default_user = "server"

        # check if there is a root user, add it
        with Session.begin() as session:
            if not session.execute(select(Users).where(user="root").exists()):
                self.create_root_session()

    def create_db(self):
        logging.info("Create database")
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)

    def create_root_session(self) -> None:
        """
        Create root session
        :return: None
        """
        pwd: str = get_root_pwd()
        hash_pwd: bytes = get_hash(pwd)
        self.add_user("root", hash_pwd.hex(), "root", "system")

    def add_user(
        self,
        username: str,
        password: str,
        role: str,
        created_by: str,
        contact: str = "",
    ):
        with Session.begin() as session:
            user = Users(
                user=username,
                key=password,
                description=role,
                created_by=created_by,
                time=datetime.datetime.now(),
                contact=contact,
            )
            session.add(user)
            try:
            except IntegrityError as e:
                session.rollback()
                raise DBException from e

    def add_log(self, user: str, action: str, project_slug: str, connect: str):
        with Session.begin() as session:
            log = Logs(
                user=user,
                project=project_slug,
                action=action,
                connect=connect,
                time=datetime.datetime.now(),
            )
            session.add(log)

    def get_logs(self, username: str, project_slug: str, limit: int):
        with Session.begin() as session:
            stmt = select(Logs)
            if project_slug == "all":
                stmt = stmt.filter_by(user=username)
            elif username == "all":
                stmt = stmt.filter_by(project=project_slug)
            stmt = stmt.order_by(Logs.time.desc()).limit(limit)

        return [
            {
                "id": log.id,
                "time": log.time,
                "user": log.user,
                "project": log.project,
                "action": log.action,
                "connect": log.connect,
            }
            for log in session.scalars(stmt)
        ]

    def get_project(self, project_slug: str):
        with Session.begin() as session:
            try:
                project = session.scalar_one(
                    select(Projects).filter_by(project_slug=project_slug)
                )
                return project
            except NoResultFound as e:
                raise DBException from e

    def add_project(self, project_slug: str, parameters: dict, username: str):
        with Session.begin() as session:
            project = Projects(
                project_slug=project_slug,
                parameters=json.dumps(parameters),
                time_created=datetime.datetime.now(),
                time_modified=datetime.datetime.now(),
                user=username,
            )
            session.add(project)
        logging.info("CREATE PROJECT at %", datetime.datetime.now())

    def update_project(self, project_slug: str, parameters: dict):
        with Session.begin() as session:
            session.execute(
                update(Projects)
                .filter_by(project_slug=project_slug)
                .values(
                    time_modified=datetime.datetime.now(),
                    parameters=json.dumps(parameters),
                )
            )

    def existing_projects(self) -> list:
        with Session.begin() as session:
            projects = session.scalars(select(Projects)).all()
            return [project.project_slug for project in projects]

    def add_token(self, token: str, status: str):
        with Session.begin() as session:
            token = Tokens(
                token=token, status=status, time_created=datetime.datetime.now()
            )
            session.add(token)

    def get_token_status(self, token: str) -> str:
        with Session.begin() as session:
            try:
                token_status = session.scalar_one(
                    select(Tokens.status).filter_by(token=token)
                )
                return token_status
            except NoResultFound as e:
                raise DBException from e

    def revoke_token(self, token: str):
        with Session.begin() as session:
            session.execute(
                update(Tokens)
                .filter_by(token=token)
                .values(time_revoked=datetime.datetime.now(), status="revoked")
            )

    def add_scheme(self, project_slug: str, name: str, labels: list, username: str):
        with Session.begin() as session:
            params = json.dumps({"labels": labels, "codebook": None})
            scheme = Schemes(
                project=project_slug,
                name=name,
                params=params,
                user=username,
                time_created=datetime.datetime.now(),
                time_modified=datetime.datetime.now(),
            )
            session.add(scheme)

    def update_scheme_labels(self, project_slug: str, name: str, labels: list):
        """
        Update the labels in the database
        """
        with Session.begin() as session:
            scheme = session.scalar_one(
                select(Schemes).filter_by(project=project_slug, name=name)
            )
            params = json.loads(scheme.params)
            params["labels"] = labels
            scheme.params = json.dumps(params)
            scheme.time_modified = datetime.datetime.now()

    def update_scheme_codebook(self, project_slug: str, scheme: str, codebook: str):
        """
        Update the codebook in the database
        """
        logging.info("update_scheme_codebook", project_slug, scheme, codebook)

        with Session.begin() as session:
            try:
                project_scheme = session.scalar_one(
                    select(Schemes).filter_by(project=project_slug, name=scheme)
                )
                params = json.loads(project_scheme.params)
                params["codebook"] = codebook
                project_scheme.params = json.dumps(params)
                project_scheme.time_modified = datetime.datetime.now()
            except NoResultFound as e:
                logging.warning("No scheme found")
                raise DBException from e
            except JSONDecodeError as e:
                logging.warning("Unable to parse codebook scheme: %", e)
                raise DBException from e

    def get_scheme_codebook(self, project_slug: str, name: str):
        with Session.begin() as session:
            try:
                scheme = session.scalar_one(
                    select(Schemes).filter_by(project=project_slug, name=name)
                )
                return {
                    "codebook": json.loads(scheme.params)["codebook"],
                    "time": str(scheme.time_modified),
                }
            except NoResultFound as e:
                logging.warning("No scheme found")
                raise DBException from e
            except JSONDecodeError as e:
                logging.warning("Unable to parse codebook scheme: %", e)
                return None

    def delete_project(self, project_slug: str):
        with Session.begin() as session:
            session.query(Projects).filter(
                Projects.project_slug == project_slug
            ).delete()
            session.query(Schemes).filter(Schemes.project == project_slug).delete()
            session.query(Annotations).filter(
                Annotations.project == project_slug
            ).delete()
            session.query(Auths).filter(Auths.project == project_slug).delete()
            session.query(Generations).filter(
                Generations.project == project_slug
            ).delete()
            session.query(Logs).filter(Logs.project == project_slug).delete()
            session.query(Features).filter(Features.project == project_slug).delete()
            session.query(Models).filter(Models.project == project_slug).delete()

    def add_generated(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
        prompt: str,
        answer: str,
    ):
        with Session.begin() as session:
            generation = Generations(
                user=user,
                time=datetime.datetime.now(),
                project=project_slug,
                element_id=element_id,
                endpoint=endpoint,
                prompt=prompt,
                answer=answer,
            )
            session.add(generation)

    def get_generated(self, project_slug: str, username: str, n_elements: int = 10):
        """
        Get elements from generated table by order desc
        """
        with Session.begin() as session:
            generated = session.scalars(
                select(Generations)
                .filter_by(project=project_slug, user=username)
                .order_by(Generations.time.desc())
                .limit(n_elements)
            ).all
            return [
                [el.time, el.element_id, el.prompt, el.answer, el.endpoint]
                for el in generated
            ]

    def get_distinct_users(self, project_slug: str, timespan: int | None):
        with Session.begin() as session:
            stmt = select(Annotations.user).filter_by(project=project_slug)
            if timespan is not None:
                time_threshold = datetime.datetime.now() - datetime.timedelta(
                    seconds=timespan
                )
                stmt = stmt.where(Annotations.time > time_threshold)
            stmt = stmt.distinct().all()
            return [u.user for u in session.scalars(stmt)]

    def get_current_users(self, timespan: int = 600):
        with Session.begin() as session:
            time_threshold = datetime.datetime.now() - datetime.timedelta(
                seconds=timespan
            )
            users = session.scalars(
                select(Logs.user).where(Logs.time > time_threshold).distinct()
            ).all()
            return [u.user for u in users]

    def get_project_auth(self, project_slug: str):
        with Session.begin() as session:
            auth = session.scalars(select(Auths).filter_by(project=project_slug)).all()
            return {el.user: el.status for el in auth}

    def add_auth(self, project_slug: str, user: str, status: str):
        with Session.begin() as session:
            try:
                auth = session.scalar_one(Auths).where(
                    Auths.project == project_slug, Auths.user == user
                )
                auth.status = status
            except NoResultFound:
                auth = Auths(project=project_slug, user=user, status=status)
                session.add(auth)
            else:

    def delete_auth(self, project_slug: str, user: str):
        with Session.begin() as session:
            session.execute(delete(Auths).filter_by(project=project_slug, user=user))

    def get_user_projects(self, username: str):
        with Session.begin() as session:
            result = session.scalars(
                select(
                    Auths.project,
                    Auths.status,
                    Projects.parameters,
                    Projects.user,
                    Projects.time_created,
                )
                .join(Projects, Auths.project == Projects.project_slug)
                .where(Auths.user == username)
            ).all()
            return [row for row in result]

    def get_user_auth(self, username: str, project_slug: str | None = None):
        with Session.begin() as session:
            stmt = select(Auths.user, Auths.status).filter_by(user=username)

            if project_slug is not None:
                stmt = stmt.filter_by(project=project_slug)
            return [[row[0], row[1]] for row in session.scalars(stmt).all()]

    def get_users_created_by(self, username: str):
        """
        get users created by *username*
        """
        with Session.begin() as session:
            stmt = select(Users.user, Users.contact)
            if username != "all":
                stmt = stmt.filter_by(created_by=username)
            stmt = stmt.distinct().all()
            return {
                row.user: {"contact": row.contact}
                for row in session.scalars(stmt).all()
            }

    def delete_user(self, username: str):
        with Session.begin() as session:
            session.execute(delete(Users).filter_by(user=username))

    def get_user(self, username: str) -> UserInDBModel:
        with Session.begin() as session:
            try:
                user = session.scalar_one(
                    select(Users.user, Users.key, Users.description).filter_by(
                        user = username
                    )
                )
                return UserInDBModel(username=user[0], hashed_password=user[1], status=user[2])
            except NoResultFound as e:
                raise DBException from e

    def change_password(self, username: str, password: str):
        with Session.begin() as session:
            session.execute(update(Users).filter_by(user = username).values(
            key = password))

    def get_scheme_elements(self, project_slug: str, scheme: str, dataset: list[str]):
        """
        Get last annotation for each element id for a project/scheme
        """
        with Session.begin() as session:
            results = session.scalars(select(
                    Annotations.element_id,
                    Annotations.annotation,
                    Annotations.user,
                    Annotations.time,
                    Annotations.comment,
                    func.max(Annotations.time),
                )
                .where(
                    Annotations.scheme == scheme,
                    Annotations.project == project_slug,
                    Annotations.dataset.in_(dataset),
                )
                .group_by(Annotations.element_id)
                .order_by(func.max(Annotations.time).desc())
            )

        # Execute the query and fetch all results
        return [
            [row.element_id, row.annotation, row.user, row.time, row.comment]
            for row in results
        ]

    def get_coding_users(self, scheme: str, project_slug: str):
        with Session.begin() as session:
            return session.scalars(select(Annotations.user)
                .filter(Annotations.project == project_slug, Annotations.scheme == scheme)
                .distinct()).all()

    def get_recent_annotations(
        self, project_slug: str, user: str, scheme: str, limit: int
    ):
        with Session.begin() as session:
            stmt = select(Annotations.element_id).filter_by(
                        project = project_slug,
                        scheme =scheme,
                        dataset = "train",
                    )
            if user != "all":
                stmt = stmt.filter_by(user = user)

            stmt = stmt                  .order_by(Annotations.time.desc()) .limit(limit) .distinct()
            return [u[0] for u in session.scalars(stmt).all()]

    def get_annotations_by_element(
        self, project_slug: str, scheme: str, element_id: str, limit: int = 10
    ):
        with Session.begin() as session:
                annotations = session.scalars(select(
                    Annotations.annotation,
                    Annotations.dataset,
                    Annotations.user,
                    Annotations.time,
                )
                .filter(
                    Annotations.project == project_slug,
                    Annotations.scheme == scheme,
                    Annotations.element_id == element_id,
                )
                .order_by(Annotations.time.desc())
                .limit(limit)).all()
        return [[a.annotation, a.dataset, a.user, a.time] for a in annotations]

    def add_annotations(
        self,
        dataset: str,
        user: str,
        project_slug: str,
        scheme: str,
        elements: list[
            dict
        ],  # [{"element_id": str, "annotation": str, "comment": str}]
    ):
        with Session.begin() as session:
            for e in elements:
                annotation = Annotations(
                    time=datetime.datetime.now(),
                    dataset=dataset,
                    user=user,
                    project=project_slug,
                    element_id=e["element_id"],
                    scheme=scheme,
                    annotation=e["annotation"],
                    comment=e["comment"],
                )
                session.add(annotation)

    def add_annotation(
        self,
        dataset: str,
        user: str,
        project_slug: str,
        element_id: str,
        scheme: str,
        annotation: str,
        comment: str = "",
    ):
        with Session.begin() as session:
            annotation = Annotations(
                time=datetime.datetime.now(),
                dataset=dataset,
                user=user,
                project=project_slug,
                element_id=element_id,
                scheme=scheme,
                annotation=annotation,
                comment=comment,
            )
            session.add(annotation)

    def available_schemes(self, project_slug: str):
        with Session.begin() as session:
            schemes = session.scalars(select(Schemes.name, Schemes.params)
                .filter(Schemes.project == project_slug)
                .distinct())                .all()
            return [
                {
                    "name": s.name,
                    "labels": json.loads(s.params)["labels"],
                    "codebook": json.loads(s.params)["codebook"],
                }
                for s in schemes
            ]

    def delete_scheme(self, project_slug: str, name: str):
        with Session.begin() as session:
            session.execute(delete(Schemes).filter_by(
                name = name, project = project_slug
            ))

    def get_table_annotations_users(self, project_slug: str, scheme: str):
        with Session.begin() as session:
            subquery = (
                select(
                    Annotations.id,
                    Annotations.user,
                    func.max(Annotations.time).label("last_timestamp"),
                )
                .where(Annotations.project == project_slug, Annotations.scheme == scheme)
                .group_by(Annotations.element_id, Annotations.user)
                .subquery()
            )
            query = select(
                Annotations.element_id,
                Annotations.annotation,
                Annotations.user,
                Annotations.time,
            ).join(subquery, Annotations.id == subquery.c.id)

            results = session.execute(query).fetchall()
            return [[row.element_id, row.annotation, row.user, row.time] for row in results]

    # feature management

    def add_feature(
        self,
        project: str,
        kind: str,
        name: str,
        parameters: str,
        user: str,
        data: str | None = None,
    ):
        with Session.begin() as session:
            feature = Features(
                project=project,
                time=datetime.datetime.now(),
                kind=kind,
                name=name,
                parameters=parameters,
                user=user,
                data=data,
            )
            session.add(feature)

    def delete_feature(self, project: str, name: str):
        with Session.begin() as session:
            session.execute(delete(Features).filter_by(
                name = name, project = project
            ))

    def get_feature(self, project: str, name: str):
        with Session.begin() as session:
            try:
                return session.scalar_one(select(Features).filter_by(name = name, project = project))
            except NoResultFound as e:
                raise DBException from e

    def get_project_features(self, project: str):
        with Session.begin() as session:
            features = session.scalars(select(Features).filter_by(project == project)).all()
            return {
                i.name: {
                    "time": i.time.strftime("%Y-%m-%d %H:%M:%S"),
                    "kind": i.kind,
                    "parameters": json.loads(i.parameters),
                    "user": i.user,
                    "data": json.loads(i.data),
                }
                for i in features
            }

    def add_model(
        self,
        kind: str,
        project: str,
        name: str,
        user: str,
        status: str,
        scheme: str,
        params: dict,
        path: str,
    ) -> None:
        with Session.begin() as session:
            # test if the name does not exist
            already_exists = session.execute(select(Models).filter_by(name = name)).exists()
            if already_exists:
                raise DBException('Model already exist')

            model = Models(
                project=project,
                time=datetime.datetime.now(),
                kind=kind,
                name=name,
                user=user,
                parameters=json.dumps(params),
                scheme=scheme,
                status=status,
                path=path,
            )
            session.add(model)

            logging.info("available %s", self.available_models(project))

    def change_model_status(self, project: str, name: str, status: str = "trained"):
        with Session.begin() as session:
            session.execute(
                update(Models)
                .filter_by(name = name, project = project)
                .values(status=status)
            )

    def available_models(self, project: str):
        with Session.begin() as session:
            models = session.scalars(select(Models.name, Models.parameters, Models.path, Models.scheme)
                .filter_by(
                    project = project,
                    status = "trained",
                )
                .distinct()).all()
            return [
                {
                    "name": m.name,
                    "scheme": m.scheme,
                    "path": m.path,
                    "parameters": json.loads(m.parameters),
                }
                for m in models
            ]

    def model_exists(self, project: str, name: str):
        with Session.begin() as session:
            return session.execute(select(Models)
                .filter(name = name, project = project)).exists()

    def delete_model(self, project: str, name: str):
        with Session.begin() as session:
            # delete the model
            session.execute(delete(Models).filter_by(
                name = name, project = project
            ))

    def get_model(self, project: str, name: str):
        with Session.begin() as session:
            return session.scalar_one(select(Models)
                .filter_by(name = name, project = project)
            )

    def rename_model(self, project: str, old_name: str, new_name: str) -> None:
        with Session.begin() as session:
            # test if the name does not exist
            exists = session.execute(select(Models)
                .filter_by(name = new_name, project = project)
                .exists()
            )
            if exists:
                raise DBException("The new name already exists")

            # get and rename
            session.execute(update(Models)
                .filter_by(name = old_name, project = project)
                .values(name = new_name)) #path = model.path.replace(old_name, new_name)

    def set_model_params(self, project: str, name: str, flag: str):
        with Session.begin() as session:
            session.execute(update(Models)
                .filter_by(name = name, project = project)
                .values(parameters= func.json_set(Models.parameters, "$.flag", flag))
            )
