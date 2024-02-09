import project
from pathlib import Path

def test_session_parameters():
    """
    Test session class
    """
    s = project.Session()
    assert type(s.max_file_size) == int
    assert isinstance(s.path, Path)
    assert type(s.features_file) == str
    assert type(s.labels_file) == str
    assert type(s.data_file) == str

def test_server():
    """
    Test server class
    """
    s = project.Server()
    assert type(s.projects) == dict