"""
Define the orchestrator and launch the instance
"""

import asyncio
import logging
import os
import secrets
import shutil
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import]
import psutil  # type: ignore[import]
from fastapi.encoders import jsonable_encoder
from jose import jwt  # type: ignore[import]

from activetigger import __version__
from activetigger.config import config
from activetigger.datamodels import (
    LMComputing,
    ProjectBaseModel,
    ProjectModel,
    ProjectSummaryModel,
    ServerStateModel,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.functions import get_dir_size, get_gpu_memory_info, slugify
from activetigger.messages import Messages
from activetigger.project import Project
from activetigger.queue import Queue
from activetigger.users import Users

logger = logging.getLogger("server")


class Orchestrator:
    """
    Server to manage the API
    Mono-process
    """

    db_name: str
    features_file: str
    data_all: str
    train_file: str
    test_file: str
    default_user: str
    jwt_algorithm: str
    n_workers_cpu: int
    n_workers_gpu: int
    starting_time: float
    path: Path
    path_models: Path
    db: Path
    projects: dict[str, Project]
    db_manager: DatabaseManager
    queue: Queue
    users: Users
    messages: Messages
    max_projects: int
    project_creation_ongoing: dict[str, Project]

    def __init__(self) -> None:
        """
        Start the server
        """
        self.starting_time = time.time()

        self.max_projects = config.max_loaded_projects
        self.data_all = config.data_all
        self.features_file = config.features_file
        self.train_file = config.train_file
        self.test_file = config.test_file
        self.default_user = config.default_user
        self.jwt_algorithm = config.jwt_algorithm
        self.n_workers_cpu = config.n_workers_cpu
        self.n_workers_gpu = config.n_workers_gpu

        # Define path
        self.path = Path(config.data_path + "/projects")
        self.path_models = Path(config.data_path + "/models")

        # create directories parent/static/models
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path.joinpath("static")).mkdir(parents=True, exist_ok=True)
        self.path_models.mkdir(exist_ok=True)

        # attributes of the server
        self.db_manager = DatabaseManager()
        self.queue = Queue(
            nb_workers_cpu=self.n_workers_cpu,
            nb_workers_gpu=self.n_workers_gpu,
        )
        self.users = Users(self.db_manager)
        self.messages = Messages(self.db_manager)
        self.projects = {}

        # timestamp of project creation
        self.project_creation_ongoing = {}

        # update the projects asynchronously
        self._running = True
        self._update_task = asyncio.create_task(self._update(timeout=config.update_timeout))

        # create the demo project if not existing
        try:
            if "demo" not in self.existing_projects():
                self.create_demo_project("demo", "demo")
        except Exception as e:
            print(f"Error while creating demo project: {e}")

        # logging
        logging.basicConfig(
            filename=self.path.joinpath("log_server.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.server_state = self.get_server_state()

    def __del__(self):
        """
        Close the server
        """
        print("Ending the server")
        logger.error("Disconnect server")
        if self._update_task:
            self._update_task.cancel()
        self._running = False
        del self.queue
        print("Server off")

    def reset(self):
        """
        Erase the waiting queue and projects in memory
        """
        self.queue.restart()
        self.projects = {}

    def starting_project_creation(self, project: ProjectBaseModel, username: str) -> str:
        """
        Start the project creation
        """
        project_slug = self.check_project_name(project.project_name)
        print("Starting project creation", project.project_name)
        # create a object project and start creation
        p = Project(
            project_slug,
            self.queue,
            self.db_manager,
            path_models=self.path_models,
        )
        p.start_project_creation(
            params=project,
            username=username,
            path=self.path,
        )
        # add it to the orchestrator
        self.project_creation_ongoing[project_slug] = p
        return project_slug

    async def _update(self, timeout: int = 1, project_lifetime: int = 7200) -> None:
        """
        Update each project in memory every X seconds.
        - loaded project
        - project in creation
        Remove projects that are older than project_lifetime seconds.
        """
        try:
            while self._running:
                print("update orchestrator - projets in memory:", len(self.projects))
                self.queue.clean_old_processes()
                timer = time.time()

                # loop on loaded projects
                to_del = []
                for p, project in self.projects.items():
                    try:
                        # if project existing since one day, remove it from memory
                        if (timer - project.starting_time) > project_lifetime:
                            to_del.append(p)
                            continue
                        project.update_processes()
                    except Exception as e:
                        print(f"Error while updating project {p}: {e}")
                        to_del.append(p)
                for p in to_del:
                    del self.projects[p]

                # loop on creating projects
                to_del = []
                for p, project in self.project_creation_ongoing.items():
                    try:
                        project.update_processes()
                        if project.status == "created":
                            to_del.append(p)
                    except Exception as e:
                        print(f"Error while updating project {p}: {e}")
                        to_del.append(p)
                for p in to_del:
                    del self.project_creation_ongoing[p]

                # update the information on the state of the project
                self.server_state = self.get_server_state()
                await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            print("Update task cancelled.")
        finally:
            print("Update task finished.")

    def log_action(
        self,
        user: str,
        action: str,
        project: str = "general",
        connect="not implemented",
    ) -> None:
        """
        Log action in the database
        """
        self.db_manager.logs_service.add_log(user, action, project, connect)
        logger.info("%s from %s in project %s", action, user, project)

    def get_logs(self, project_slug: str, limit: int, partial: bool = True) -> pd.DataFrame:
        """
        Get logs for a user/project
        project_slug: project slug or "all"
        """
        logs = self.db_manager.logs_service.get_logs("all", project_slug, limit)
        df = pd.DataFrame(logs, columns=["id", "time", "user", "project", "action", "NA"])
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

    def get_server_state(self) -> ServerStateModel:
        """
        Build server state
        """
        # active projects in the orchestrator
        active_projects = {
            p: [
                {"unique_id": c.unique_id, "user": c.user, "kind": c.kind, "time": c.time}
                for c in self.projects[p].computing
            ]
            for p in self.projects
        }

        # running processes in the queue
        queue = {
            i.unique_id: i.model_dump()
            for i in self.queue.state()
            if i.state in ["pending", "running"]
        }

        # server state
        cpu = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage("/")
        at_memory = get_dir_size(config.data_path + "/projects")

        return ServerStateModel(
            version=__version__,
            active_projects=active_projects,
            queue=queue,
            gpu=get_gpu_memory_info(),
            cpu={"proportion": cpu, "total": cpu_count},
            memory={
                "proportion": memory_info.percent,
                "total": memory_info.total / (1024**3),
                "available": memory_info.available / (1024**3),
            },
            disk={
                "activetigger": at_memory,
                "proportion": disk_info.percent,
                "total": disk_info.total / (1024**3),
            },
            mail_available=self.messages.mail_available,
        )

    def get_auth_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get projects authorized for the user
        """
        projects_auth = self.users.get_auth_projects(username)
        projects = []
        for i in list(reversed(projects_auth)):
            # get the project slug
            project_slug = i[0]
            user_right = i[1]
            parameters = self.db_manager.projects_service.get_project(project_slug)
            if parameters is None:
                continue
            parameters = ProjectModel(**parameters["parameters"])
            created_by = i[3]
            created_at = i[4].strftime("%Y-%m-%d %H:%M:%S")
            try:
                size = round(get_dir_size(config.data_path + "/projects/" + i[0]), 1)
            except Exception as e:
                print(e)
                size = 0.0
            last_activity = self.db_manager.logs_service.get_last_activity_project(i[0])

            # create the project summary model
            projects.append(
                ProjectSummaryModel(
                    project_slug=project_slug,
                    user_right=user_right,
                    parameters=parameters,
                    created_by=created_by,
                    created_at=created_at,
                    size=size,
                    last_activity=last_activity,
                )
            )
        return projects

    def get_project_params(self, project_slug: str) -> ProjectModel | None:
        """
        Get project params from database
        """
        existing_project = self.db_manager.projects_service.get_project(project_slug)
        if existing_project:
            return ProjectModel(**existing_project["parameters"])
        else:
            return None

    def exists(self, project_name: str) -> bool:
        """
        Test if a project exists in the database
        with a sluggified form (to be able to use it in URL)
        """
        return slugify(project_name) in self.existing_projects()

    def check_project_name(self, project_name: str) -> str:
        """
        Check if a project name is valid
        """
        project_slug = slugify(project_name)
        if self.exists(project_slug):
            raise Exception("This project already exists")
        if project_slug == "":
            raise Exception("The project name is not valid - empty")
        return project_slug

    def create_access_token(self, data: dict, expires_min: int = 60):
        """
        Create access token
        """
        # create the token
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, config.secret_key, algorithm=self.jwt_algorithm)

        # add it in the database as active
        self.db_manager.projects_service.add_token(encoded_jwt, "active")

        # return it
        return encoded_jwt

    def revoke_access_token(self, token) -> None:
        """
        Revoke existing access token
        """
        self.db_manager.projects_service.revoke_token(token)
        return None

    def decode_access_token(self, token: str):
        """
        Decode access token
        """
        # get status
        try:
            status = self.db_manager.projects_service.get_token_status(token)
        except DBException as e:
            raise Exception from e

        if status != "active":
            raise Exception("Token is invalid")

        # decode payload
        payload = jwt.decode(token, config.secret_key, algorithms=[self.jwt_algorithm])
        return payload

    def start_project(self, project_slug: str) -> None:
        """
        Load project in server
        """
        if not self.exists(project_slug):
            raise Exception("This project does not exist")

        try:
            self.projects[project_slug] = Project(
                project_slug, self.queue, self.db_manager, path_models=self.path_models
            )
        except Exception as e:
            raise Exception(
                f"Error while loading project {project_slug}: {e} - {traceback.format_exc()}"
            ) from e

    def stop_project(self, project_slug: str) -> None:
        """
        Stop a project
        """
        if project_slug not in self.projects:
            return None
        del self.projects[project_slug]

    def stop_process(self, process_id: str, username: str) -> None:
        """
        Stop a specific process
        """
        self.queue.kill(process_id)
        self.log_action(username, f"KILL PROCESS: {process_id}", "all")

    def stop_user_processes(self, kind: str | list, username: str):
        """
        Stop all the processes of a user
        """

        # kill all the process of a user
        if kind == "all":
            kind = ["train_bert", "predict_bert", "generation", "feature", "bertopic"]
        if isinstance(kind, str) and kind != "all":
            kind = [kind]

        # get all processes associated to the user for the specified kind
        processes = {p: self.projects[p].get_process(kind, username) for p in self.projects}

        # kill all processes associated
        for project in processes:
            for process in processes[project]:
                self.queue.kill(process.unique_id)
                if process.kind == "train_bert":
                    process = cast(LMComputing, process)
                    self.db_manager.language_models_service.delete_model(
                        project, process.model_name
                    )

    def set_project_parameters(self, project: ProjectModel, username: str) -> None:
        """
        Update project parameters in the DB
        """
        # get project
        existing_project = self.db_manager.projects_service.get_project(project.project_slug)

        if existing_project:
            # Update the existing project
            self.db_manager.projects_service.update_project(
                project.project_slug, jsonable_encoder(project)
            )
        else:
            # Insert a new project
            self.db_manager.projects_service.add_project(
                project.project_slug, jsonable_encoder(project), username
            )

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        return self.db_manager.projects_service.existing_projects()

    def delete_project(self, project_slug: str) -> None:
        """
        Delete a project
        """

        # test if the project exists
        if not self.exists(project_slug):
            raise Exception("This project does not exist")

        # load the project
        if project_slug in self.projects:
            project = self.projects[project_slug]
        else:
            self.start_project(project_slug)
            project = self.projects[project_slug]

        # delete the project
        try:
            project.delete()
        except Exception as e:
            raise Exception from e

        # clean current memory
        if project_slug in self.projects:
            del self.projects[project_slug]

    def clean_unfinished_project(
        self, project_slug: str | None = None, project_name: str | None = None
    ) -> None:
        """
        Clean a project that is not loadable
        """

        if project_slug is not None:
            project_slug_verif = project_slug
        elif project_name is not None:
            project_slug_verif = slugify(project_name)
        else:
            raise Exception("No project specified")

        # remove from database
        self.db_manager.projects_service.delete_project(project_slug_verif)

        # remove from the server
        if project_slug_verif and project_slug_verif != "":
            shutil.rmtree(self.path.joinpath(project_slug_verif), ignore_errors=True)

            ## remove static files
            if Path(f"{config.data_path}/projects/static/{project_slug_verif}").exists():
                shutil.rmtree(f"{config.data_path}/projects/static/{project_slug_verif}")

    def create_demo_project(self, project_name: str, username: str) -> None:
        """
        Create a demo project for a specific user
        """
        # TODO : put those elements in the config file
        path_data = Path("../frontend/public/dataset_test.csv")
        col_id = "id"
        col_text = "sentence"
        col_label = "label_agg"

        if not path_data.exists():
            raise Exception("The demo dataset is not available")

        # create name of the project and the directory
        project_name = project_name
        project_slug = slugify(project_name)
        project_path = self.path.joinpath(project_slug)
        os.makedirs(project_path, exist_ok=True)

        # create and save a toy dataset
        df = pd.read_csv(path_data)
        df.to_csv(project_path.joinpath("dataset.csv"), index=False)

        # parameters of the project and creation
        project = ProjectBaseModel(
            project_name=project_name,
            language="en",
            filename="dataset.csv",
            col_id=col_id,
            cols_text=[col_text],
            cols_label=[col_label],
            n_train=1000,
            n_test=500,
        )
        self.starting_project_creation(project, username)

    def reset_password(self, mail: str) -> None:
        """
        Reset password for a user with the given email
        """
        # Check if mail is connected to a user
        user_name = self.db_manager.users_service.get_user_by_mail(mail)

        # Generate a random password
        new_password = secrets.token_hex(16)

        # Send the mail to the user with the new password
        self.messages.send_mail_reset_password(user_name, mail, new_password)

        # Update the user's password in the database
        self.users.force_change_password(user_name, new_password)


# launch the instance
orchestrator = Orchestrator()
