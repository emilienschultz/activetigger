import pytest
from pathlib import Path
from fastapi import UploadFile
import server
import os
import shutil


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
    s = server.Server()
    yield s


#############
# Test Queue#
#############


def test_create_queue():
    from server import Queue
    import concurrent.futures

    queue = Queue(2)

    assert isinstance(queue.executor, concurrent.futures.ProcessPoolExecutor)
    assert len(queue.current) == 0

    queue.recreate_executor()


def test_add_job_queue():
    from server import Queue

    def dummy_func(x):
        return x * 2

    queue = Queue(2)

    num = queue.add("test", dummy_func, {"x": 2})

    assert isinstance(num, str)


def test_shutdown_queue():
    from server import Queue

    queue = Queue(2)
    queue.close()


# @pytest.fixture
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


def test_server(start_server):
    """
    Run server and test its components
    """
    s = start_server
    # print(type(s))
    # assert isinstance(s.projects, dict)
    # assert s.db.exists()


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
