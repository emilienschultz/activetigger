# from fastapi.testclient import TestClient
# from api import app
# import server as server_object
# import pytest


# @pytest.fixture
# def root_pwd():
#     return "emilien"


# @pytest.fixture
# def running_server(monkeypatch, root_pwd):
#     monkeypatch.setattr("builtins.input", lambda _: root_pwd)
#     yield server_object.Server()


# client = TestClient(app)
# server = running_server


# def test_read_main():
#     """
#     Test main route
#     """
#     response = client.get("/")
#     assert response.status_code == 200


# def test_get_token(root_pwd):
#     """
#     Test authentification
#     """
#     response = client.post("/token", data={"username": "root", "password": root_pwd})
#     assert response.status_code == 200
