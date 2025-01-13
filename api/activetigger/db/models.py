import datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON}


class Projects(Base):
    __tablename__: str = "projects"

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
    user: Mapped["Users"] = relationship("Users")
    schemes: Mapped[list["Schemes"]] = relationship(
        "Schemes", cascade="all, delete-orphan", back_populates="project"
    )
    annotations: Mapped[list["Annotations"]] = relationship(
        "Annotations", cascade="all, delete-orphan", back_populates="project"
    )
    auths: Mapped[list["Auths"]] = relationship(
        "Auths", cascade="all, delete-orphan", back_populates="project"
    )
    logs: Mapped[list["Logs"]] = relationship(
        "Logs", cascade="all, delete-orphan", back_populates="project"
    )
    generations: Mapped[list["Generations"]] = relationship(
        "Generations", cascade="all, delete-orphan", back_populates="project"
    )
    features: Mapped[list["Features"]] = relationship(
        "Features", cascade="all, delete-orphan", back_populates="project"
    )
    models: Mapped[list["Models"]] = relationship(
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
    projects: Mapped[list[Projects]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


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
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="schemes")
    models: Mapped[list["Models"]] = relationship()
    name: Mapped[str]
    params: Mapped[dict[str, Any]]


class Annotations(Base):
    __tablename__ = "annotations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    dataset: Mapped[str]
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="annotations")
    element_id: Mapped[str]
    scheme_id: Mapped[int] = mapped_column(ForeignKey("schemes.id"))
    scheme: Mapped[Schemes] = relationship()
    annotation: Mapped[str]
    comment: Mapped[str | None] = mapped_column(Text)


class Auths(Base):
    __tablename__ = "auth"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="auths")
    status: Mapped[str]
    created_by: Mapped[str | None]


class Logs(Base):
    __tablename__ = "logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="logs")
    action: Mapped[str | None]
    connect: Mapped[str | None]


class Tokens(Base):
    __tablename__ = "tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time_created: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    token: Mapped[str]
    status: Mapped[str]
    time_revoked: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))


class Generations(Base):
    __tablename__ = "generations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    time: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
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
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="features")
    name: Mapped[str]
    kind: Mapped[str]
    parameters: Mapped[dict[str, Any]]
    data: Mapped[dict[str, Any]]


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
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    user: Mapped[Users] = relationship()
    project_id: Mapped[str] = mapped_column(ForeignKey("projects.project_slug"))
    project: Mapped[Projects] = relationship(back_populates="models")
    scheme_id: Mapped[int] = mapped_column(ForeignKey("schemes.id"))
    scheme: Mapped[Schemes] = relationship(back_populates="models")
    kind: Mapped[str]
    name: Mapped[str]
    parameters: Mapped[dict[str, Any]]
    path: Mapped[str]
    status: Mapped[str]
    statistics: Mapped[str | None]
    test: Mapped[str | None]
