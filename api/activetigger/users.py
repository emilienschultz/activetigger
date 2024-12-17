import os
from pathlib import Path

import yaml

from activetigger.datamodels import UserInDBModel
from activetigger.db import DBException, DatabaseManager
from activetigger.functions import compare_to_hash, get_hash


class UserException(Exception):
    pass


class Users:
    """
    Managers users
    """

    db_manager: DatabaseManager

    def __init__(
        self,
        db_manager: DatabaseManager,
        file_users: str = "add_users.yaml",
    ):
        """
        Init users references
        """
        self.db_manager = db_manager

        # add users if add_users.yaml exists
        if Path(file_users).exists():
            existing = self.existing_users()
            with open("add_users.yaml") as f:
                add_users = yaml.safe_load(f)
            for user, password in add_users.items():
                if user not in existing:
                    self.add_user(user, password, "manager", "system")
                else:
                    print(f"Not possible to add {user}, already exists")
            # rename the file
            os.rename("add_users.yaml", "add_users_processed.yaml")

    def get_project_auth(self, project_slug: str):
        """
        Get user auth for a project
        """
        auth = self.db_manager.get_project_auth(project_slug)
        return auth

    def set_auth(self, username: str, project_slug: str, status: str):
        """
        Set user auth for a project
        """
        try:
            self.db_manager.add_auth(project_slug, username, status)
        except Exception as e:
            raise UserException from e

    def delete_auth(self, username: str, project_slug: str):
        """
        Delete user auth
        """
        if username == "root":
            raise UserException("Can't delete root user auth")
        self.db_manager.delete_auth(project_slug, username)

    def get_auth_projects(self, username: str) -> list:
        """
        Get user auth
        """
        auth = self.db_manager.get_user_projects(username)
        return auth

    def get_auth(self, username: str, project_slug: str = "all") -> list:
        """
        Get user auth
        Comments:
        - Either for all projects
        - Or one project
        """
        if project_slug == "all":
            auth = self.db_manager.get_user_auth(username)
        else:
            auth = self.db_manager.get_user_auth(username, project_slug)
        return auth

    def existing_users(self, username: str = "root") -> dict:
        """
        Get existing users which have been created by one user
        (except root which can't be modified)
        TODO : better rules
        """
        if username == "root":
            users = self.db_manager.get_users_created_by("all")
        else:
            users = self.db_manager.get_users_created_by(username)
        return users

    def add_user(
        self,
        name: str,
        password: str,
        role: str = "manager",
        created_by: str = "NA",
        mail: str = "NA",
    ):
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        hash_pwd = get_hash(password)
        # DB will throw an exception if user unicity constraint is violated
        try:
            self.db_manager.add_user(
                name, hash_pwd.hex(), role, created_by, contact=mail
            )
        except DBException as e:
            raise UserException from e

    def delete_user(self, user_to_delete: str, username: str):
        """
        Deleting user
        """
        # test specific rights
        if user_to_delete == "root":
            raise UserException("Can't delete root user")
        if user_to_delete not in self.existing_users():
            raise UserException("Username does not exist")
        if user_to_delete not in self.existing_users(username):
            raise UserException("You don't have the right to delete this user")

        # delete the user
        self.db_manager.delete_user(user_to_delete)

    def get_user(self, name) -> UserInDBModel:
        """
        Get user from database
        """
        try:
            return self.db_manager.get_user(name)
        except DBException as e:
            raise Exception(e)

    def authenticate_user(self, username: str, password: str) -> UserInDBModel:
        """
        User authentification
        """
        try:
            user = self.get_user(username)
        except DBException as e:
            raise Exception(e)
        if not compare_to_hash(password, user.hashed_password):
            raise Exception("Wrong password")
        return user

    def auth(self, username: str, project_slug: str):
        """
        Check auth for a specific project
        """
        user_auth = self.get_auth(username, project_slug)
        if len(user_auth) == 0:  # not associated
            return None
        return user_auth[0][1]

    def change_password(
        self, username: str, password_old: str, password1: str, password2: str
    ):
        """
        Change password for a user
        """
        if password1 != password2:
            raise UserException("Passwords don't match")
        user = self.get_user(username)
        if not compare_to_hash(password_old, user.hashed_password):
            raise UserException("Wrong password")
        hash_pwd = get_hash(password1)
        self.db_manager.change_password(username, hash_pwd.hex())
