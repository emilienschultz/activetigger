import project
from pathlib import Path

def test_session_parameters():
    """
    Test session class
    """
    s = project.Session()
    assert isinstance(s.max_file_size, int)
    assert isinstance(s.path, Path)
    assert isinstance(s.features_file, str)
    assert isinstance(s.labels_file,str)
    assert isinstance(s.data_file, str)

def test_server():
    """
    Test server class
    """
    s = project.Server()
    assert isinstance(s.projects, dict)
    assert s.db.exists()

def test_db():
    """
    Test if db is initialized
    """
    s = project.Server()
    assert isinstance(s.existing_projects(), list)

