import os
from pathlib import Path

import yaml

from activetigger.datamodels import UserInDBModel
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

    def get_project_auth(self, project_slug: str):
        """
        Get user auth for a project
        """
        auth = self.db_manager.projects_service.get_project_auth(project_slug)
        return auth

    def set_auth(self, username: str, project_slug: str, status: str):
        """
        Set user auth for a project
        """
        self.db_manager.projects_service.add_auth(project_slug, username, status)
        return {"success": "Auth added to database"}

    def delete_auth(self, username: str, project_slug: str):
        """
        Delete user auth
        """
        if username == "root":
            return {"error": "Can't delete root user auth"}
        self.db_manager.projects_service.delete_auth(project_slug, username)
        return {"success": "Auth deleted"}

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

    def existing_users(self, username: str = "root") -> dict:
        """
        Get existing users which have been created by one user
        (except root which can't be modified)
        TODO : better rules
        """
        if username == "root":
            users = self.db_manager.users_service.get_users_created_by("all")
        else:
            users = self.db_manager.users_service.get_users_created_by(username)
        return users

    def add_user(
        self,
        name: str,
        password: str,
        role: str = "manager",
        created_by: str = "NA",
        mail: str = "NA",
    ) -> bool:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist
        if name in self.existing_users():
            return {"error": "Username already exists"}
        hash_pwd = get_hash(password)
        self.db_manager.users_service.add_user(
            name, hash_pwd, role, created_by, contact=mail
        )

        return {"success": "User added to the database"}

    def delete_user(self, user_to_delete: str, username: str) -> dict:
        """
        Deleting user
        """
        # test specific rights
        if user_to_delete == "root":
            return {"error": "Can't delete root user"}
        if user_to_delete not in self.existing_users():
            return {"error": "Username does not exist"}
        if user_to_delete not in self.existing_users("root"):
            return {"error": "You don't have the right to delete this user"}

        # delete the user
        self.db_manager.users_service.delete_user(user_to_delete)

        return {"success": "User deleted"}

    def get_user(self, name) -> UserInDBModel | dict:
        """
        Get user from database
        """
        if name not in self.existing_users():
            return {"error": "Username doesn't exist"}
        user = self.db_manager.users_service.get_user(name)
        return UserInDBModel(
            username=name, hashed_password=user.key, status=user.description
        )

    def authenticate_user(
        self, username: str, password: str
    ) -> UserInDBModel | dict[str, str]:
        """
        User authentification
        """
        user = self.get_user(username)
        if not isinstance(user, UserInDBModel):
            return user
        if not compare_to_hash(password, user.hashed_password):
            return {"error": "Wrong password"}
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
            return {"error": "Passwords don't match"}
        user = self.get_user(username)
        if not isinstance(user, UserInDBModel):
            return {"error": "User doesn't exist"}
        if not compare_to_hash(password_old, user.hashed_password):
            return {"error": "Wrong password"}
        hash_pwd = get_hash(password1)
        self.db_manager.users_service.change_password(username, hash_pwd)
        return None
