import os
import shutil

import pytest
from activetigger.datamodels import ProjectDataModel
from activetigger.server import Server


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
    s = Server()
    yield s


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


@pytest.fixture
def project(start_server, new_project):
    """
    Create a project
    """
    # create project
    r = start_server.create_project(new_project, "test")
    assert "error" not in r

    # start & get project
    start_server.start_project("test")
    project = start_server.projects["test"]

    yield project

    # delete project
    r = start_server.delete_project(project.params.project_slug)
    assert "error" not in r


def test_scheme(project):

    # DEFAULT
    available = project.schemes.available()
    assert isinstance(available, dict)
    assert len(available) == 1
    assert "default" in available

    # ADD
    r = project.schemes.add_scheme("test", ["A", "B"])
    assert "error" not in r
    available = project.schemes.available()
    assert len(available) == 2
    assert "test" in available

    # UPDATE
    r = project.schemes.update_scheme("test", ["A", "B", "C"])
    assert "error" not in r
    available = project.schemes.available()
    assert len(available["test"]) == 3

    # REMOVE
    r = project.schemes.delete_scheme("test")
    assert "error" not in r
    available = project.schemes.available()
    assert len(available) == 1


# def test_add_label():
#     return None

# def test_delete_label():
#     return None


# def test_add_scheme():
#     return None


# def test_remove_scheme():
#     return None


# def get_next_element():
#     return None


# def test_add_annotation():
#     return None
