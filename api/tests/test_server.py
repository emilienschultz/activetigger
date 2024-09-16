import pytest
from pathlib import Path
from fastapi import UploadFile
from activetigger.server import Server, Queue, Users
import os
import shutil
import sqlalchemy


@pytest.fixture
def root_pwd():
    return "emilien"


@pytest.fixture
def create_and_change_directory(monkeypatch):
    """
    Fixture to create a new directory and change the current directory to it.
    """
    # create a dedicated directory to test
    new_directory = "./test_at"
    os.mkdir(new_directory)
    monkeypatch.chdir(new_directory)
    yield new_directory

    # destroy the directory after
    shutil.rmtree("../" + new_directory)


@pytest.fixture
def start_server(monkeypatch, root_pwd, create_and_change_directory):
    """
    Start a server
    """
    monkeypatch.setattr("builtins.input", lambda _: root_pwd)

    # start a session
    s = Server()
    yield s


#############
# Test Queue#
#############


def test_create_queue():
    import concurrent.futures

    queue = Queue(2)

    assert isinstance(queue.executor, concurrent.futures.ProcessPoolExecutor)
    assert len(queue.current) == 0


def test_shutdown_queue():

    queue = Queue(2)
    queue.close()


# @pytest.fixture

import time


def dummy_func(x):
    time.sleep(1)
    return x * 2


def test_add_kill_job_queue():

    queue = Queue(2)

    num = queue.add("test", dummy_func, {"x": 2})

    assert isinstance(num, str)
    assert len(queue.current) == 1

    msg = queue.kill(num)

    assert "success" in msg

    queue.close()


def test_state_queue():
    import time

    queue = Queue(2)

    assert isinstance(queue.state(), dict)

    num = queue.add("test", dummy_func, {"x": 2})
    state = queue.state()
    nb_state = queue.get_nb_active_processes()
    assert len(state) == 1
    assert nb_state == 1

    queue.close()


##############
# Test Server#
##############


def test_server(start_server):
    from pathlib import Path
    from datetime import datetime

    assert isinstance(start_server.path, Path)
    assert isinstance(start_server.path_models, Path)
    assert isinstance(start_server.path / "static", Path)
    assert isinstance(start_server.projects, dict)
    assert isinstance(start_server.users, Users)
    assert isinstance(start_server.queue, Queue)
    assert start_server.db.exists()
    assert start_server.path.exists()
    assert start_server.path_models.exists()
    assert (start_server.path / "static").exists()


def test_db_existing(start_server):

    engine = sqlalchemy.create_engine(f"sqlite:///{str(start_server.db)}")

    # Test existing tables
    expected_tables = [
        "projects",
        "schemes",
        "annotations",
        "users",
        "auth",
        "logs",
        "tokens",
        "generations",
    ]

    inspector = sqlalchemy.inspect(engine)
    actual_tables = inspector.get_table_names()

    missing_tables = [table for table in expected_tables if table not in actual_tables]

    if len(missing_tables) > 0:
        raise AssertionError(f"Les tables suivantes sont manquantes : {missing_tables}")

    # Test if root access
    from activetigger.db import User

    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    users = session.query(User).filter(User.user == "root").all()
    session.commit()
    session.close()

    assert len(users) == 1


# def project_params():
#     """
#     Parameters for a project
#     """
#     return ProjectModel(
#         project_name="pytest",
#     )


# @pytest.fixture
# def project_file():
#     """
#     Upload file for a project
#     """
#     f = UploadFile(open("../data/synth_data.csv", "rb"))
#     return f


# def test_create_delete_project(start_server, project_params, project_file):
#     """
#     Create and delete a project
#     """
#     assert start_server.create_project(project_params, project_file)
#     assert start_server.delete_project(project_params.project_name)


# def test_get_next(open_project):

#     r = open_project.get_next("default")
#     assert r["element_id"]

#     # add different ways to get next


# def test_annotate(open_project):
#     r = open_project.get_next("default")
#     r = open_project.schemes.push_tag(r["element_id"], "test", "default", "test")
#     assert r["success"]
