import logging
import os
from pathlib import Path

import yaml  # type: ignore[import]

from activetigger.datamodels import UserInDBModel, UserStatistics
from activetigger.db.manager import DatabaseManager
from activetigger.functions import compare_to_hash, get_hash


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

    def get_project_auth(self, project_slug: str) -> dict[str, str]:
        """
        Get user auth for a project
        """
        print("function")
        return self.db_manager.projects_service.get_project_auth(project_slug)

    def set_auth(self, username: str, project_slug: str, status: str) -> None:
        """
        Set user auth for a project
        """
        self.db_manager.projects_service.add_auth(project_slug, username, status)
        logging.info("Auth successfully to %s", username)

    def delete_auth(self, username: str, project_slug: str) -> None:
        """
        Delete user auth
        """
        if username == "root":
            raise Exception("Can't delete root user auth")
        self.db_manager.projects_service.delete_auth(project_slug, username)
        logging.info("Auth of user %s deleted", username)

    def get_auth_projects(self, username: str) -> list:
        """
        Get user auth
        """
        auth = self.db_manager.projects_service.get_user_projects(username)
        return auth

    def get_auth(self, username: str, project_slug: str = "all") -> list:
        """
        Get user auth
        Comments:
        - Either for all projects
        - Or one project
        """
        if project_slug == "all":
            auth = self.db_manager.projects_service.get_user_auth(username)
        else:
            auth = self.db_manager.projects_service.get_user_auth(
                username, project_slug
            )
        return auth

    def existing_users(self, username: str = "root", active: bool = True) -> dict:
        """
        Get existing users which have been created by one user
        (except root which can't be modified)
        TODO : better rules
        """
        if username == "root":
            users = self.db_manager.users_service.get_users_created_by("all", active)
        else:
            users = self.db_manager.users_service.get_users_created_by(username, active)
        return users

    def add_user(
        self,
        name: str,
        password: str,
        role: str = "manager",
        created_by: str = "NA",
        mail: str = "NA",
    ) -> None:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist, even among deactivated users
        if name in self.existing_users(active=False):
            raise Exception("Username already exists")
        hash_pwd = get_hash(password)
        self.db_manager.users_service.add_user(
            name, hash_pwd.decode("utf8"), role, created_by, contact=mail
        )

        logging.info("User added to the database")

    def delete_user(self, user_to_delete: str, username: str) -> None:
        """
        Deleting user
        """
        # test specific rights
        if user_to_delete == "root":
            raise Exception("Can't delete root user")
        if user_to_delete not in self.existing_users():
            raise Exception("Username does not exist")
        if user_to_delete not in self.existing_users("root"):
            raise Exception("You don't have the right to delete this user")

        # delete the user
        self.db_manager.users_service.delete_user(user_to_delete)

        logging.info("User %s successfully deleted", user_to_delete)

    def get_user(self, name: str) -> UserInDBModel:
        """
        Get active user from database
        """
        if name not in self.existing_users():
            raise Exception("Username doesn't exist or is deactivated")
        user = self.db_manager.users_service.get_user(name)
        return UserInDBModel(
            username=name, hashed_password=user.key, status=user.description
        )

    def authenticate_user(self, username: str, password: str) -> UserInDBModel:
        """
        User authentification
        """
        user = self.get_user(username)
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
    ) -> None:
        """
        Change password for a user
        """
        if password1 != password2:
            raise Exception("Passwords don't match")
        user = self.get_user(username)
        if not compare_to_hash(password_old, user.hashed_password):
            raise Exception("Wrong password")
        hash_pwd = get_hash(password1)
        self.db_manager.users_service.change_password(username, hash_pwd.decode("utf8"))
        return None

    def get_statistics(self, username: str) -> UserStatistics:
        """
        Get statistics for specific user
        """
        try:
            projects = {i[0]: i[1] for i in self.get_auth_projects(username)}
            return UserStatistics(username=username, projects=projects)
        except Exception as e:
            raise Exception(f"Error in getting statistics for {username}") from e
