import pytest
from pathlib import Path
from fastapi import UploadFile
import server
from datamodels import ParamsModel

@pytest.fixture
def start_server():
    """
    Start a server
    """
    yield server.Server()

@pytest.fixture
def project_params():
    """
    Parameters for a project
    """
    return ParamsModel(
        project_name = "test",
    )

@pytest.fixture
def project_file():
    """
    Upload file for a project
    """
    f = UploadFile(open("data/dataset_test.csv","rb"))
    return f

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