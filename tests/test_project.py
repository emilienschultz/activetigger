import pytest
from pathlib import Path
from fastapi import UploadFile
import server
from datamodels import ParamsModel
import os

@pytest.fixture
def start_server():
    """
    Start a server
    """
    # start a session
    s = server.Server()
    db_path = s.db
    at_path = s.path

    yield s

    # clean after the session
    if os.path.exists(db_path):
        os.remove(db_path)
    if os.path.exists(at_path):
        os.rmdir(at_path)

@pytest.fixture
def project_params():
    """
    Parameters for a project
    """
    return ParamsModel(
        project_name = "pytest",
    )

@pytest.fixture
def project_file():
    """
    Upload file for a project
    """
    f = UploadFile(open("data/dataset_test.csv","rb"))
    return f

@pytest.fixture
def open_project(start_server,project_params, project_file):
    start_server.create_project(project_params, project_file)
    start_server.start_project(project_params.project_name)
    yield start_server.projects[project_params.project_name]
    start_server.delete_project(project_params.project_name)

def test_session_parameters():
    """
    Test session class
    """
    s = server.Session()
    assert isinstance(s.max_file_size, int)
    assert isinstance(s.path, Path)
    assert isinstance(s.features_file, str)
    assert isinstance(s.labels_file,str)
    assert isinstance(s.data_file, str)

def test_server():
    """
    Test server class
    """
    s = server.Server()
    assert isinstance(s.projects, dict)
    assert s.db.exists()

def test_create_delete_project(start_server,project_params, project_file):
    """
    Create and delete a project
    """
    assert start_server.create_project(project_params, project_file)
    assert start_server.delete_project(project_params.project_name)

def test_get_next(open_project):

    r = open_project.get_next("default")
    assert r["element_id"]

    # add different ways to get next

def test_annotate(open_project):
    r = open_project.get_next("default")
    r = open_project.schemes.push_tag(r["element_id"],"test","default","test")
    assert r["success"]
