import pytest
from pathlib import Path
from activetigger.server import Server, Queue, Users
import os
import shutil
import sqlalchemy
import time

from activetigger.datamodels import ProjectDataModel


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
    from activetigger.db import Users

    Session = sqlalchemy.orm.sessionmaker(bind=engine)
    session = Session()
    users = session.query(Users).filter(Users.user == "root").all()
    session.commit()
    session.close()

    assert len(users) == 1


def test_log(start_server):
    # log action
    start_server.log_action("test", "test", "test", "test")

    # get element
    r = start_server.get_logs("test", "test", 10)

    assert len(r) > 0


@pytest.fixture
def data_file_csv():
    """
    Upload file for a project
    """
    with open("../data/synth_data.csv", "rb") as f:
        c = f.read()
    return c


@pytest.fixture
def new_project(data_file_csv):
    """
    Parameters for a project
    """
    p = ProjectDataModel(
        project_name="test",
        filename="synth_data.csv",
        col_text="text",
        col_id="index",
        col_label="label",
        n_train=100,
        n_test=100,
        language="fr",
        cols_context=["info"],
        csv=data_file_csv,
    )
    return p


def test_create_delete_project(start_server, new_project):
    """
    Create and delete a project
    """
    # create project
    r = start_server.create_project(new_project, "test")
    assert not "error" in r

    # start & get project
    start_server.start_project("test")
    project = start_server.projects["test"]

    # test properties of the project
    assert project.params.project_slug == "test"
    assert Path(project.params.dir).exists()

    # delete project
    r = start_server.delete_project(project.params.project_slug)
    assert not "error" in r

    # TODO : ADD STRATIFICATION
