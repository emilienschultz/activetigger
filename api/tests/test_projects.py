import datetime

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from activetigger.db.models import Base, Projects, Schemes
from activetigger.db.projects import ProjectsService


@pytest.fixture(scope="module")
def get_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)


@pytest.fixture
def dataset(get_session: sessionmaker[Session]):
    with get_session.begin() as session:
        now = datetime.datetime.now()
        project = Projects(
            project_slug="test_project",
            time_modified=now,
            parameters=dict(),
            time_created=now,
            user_id="test_user",
        )
        session.add(project)
        scheme = Schemes(
            project_id="test_project",
            name="test_scheme",
            params={},
            user_id="test_user",
            time_created=now,
            time_modified=now,
        )
        project.schemes.append(scheme)
        session.add(scheme)

    yield get_session


def test_delete_project(dataset: sessionmaker[Session]):
    with dataset.begin() as session:
        projects = session.scalars(select(Projects)).all()
        assert len(projects) == 1
        assert len(session.scalars(select(Schemes)).all()) == 1
        assert len(projects[0].schemes) == 1
        service = ProjectsService(dataset)
        service.delete_project("test_project")

    assert len(session.scalars(select(Projects)).all()) == 0
    assert len(session.scalars(select(Schemes)).all()) == 0
