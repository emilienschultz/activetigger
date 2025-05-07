"""
Define the orchestrator and launch the instance
"""

import asyncio
import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import psutil
import yaml  # type: ignore[import]
from cryptography.fernet import Fernet
from fastapi.encoders import jsonable_encoder
from jose import jwt
from sklearn.datasets import fetch_20newsgroups
from slugify import slugify

from activetigger import __version__
from activetigger.datamodels import (
    ProjectBaseModel,
    ProjectModel,
    ProjectSummaryModel,
    ServerStateModel,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.functions import get_dir_size, get_gpu_memory_info
from activetigger.project import Project
from activetigger.queue import Queue
from activetigger.users import Users

logger = logging.getLogger("server")


# conf deployment
ALGORITHM = "HS256"
MAX_LOADED_PROJECTS = 20
N_WORKERS_GPU = 1
N_WORKERS_CPU = 5
UPDATE_TIMEOUT = 1


class Orchestrator:
    """
    Server to manage backend
    """

    db_name: str
    features_file: str
    data_all: str
    train_file: str
    test_file: str
    default_user: str
    algorithm: str
    n_workers_cpu: int
    n_workers_gpu: int
    starting_time: float
    path: Path
    path_models: Path
    db: Path
    projects: dict
    db_manager: DatabaseManager
    queue: Queue
    users: Users
    max_projects: int

    def __init__(self, path=".", path_models="./models") -> None:
        """
        Start the server
        Use the following environment variables:
        - ACTIVETIGGER_PATH: path to the projects
        - ACTIVETIGGER_MODEL: path to
        """

        self.max_projects = MAX_LOADED_PROJECTS
        self.db_name = "activetigger.db"
        self.data_all = "data_all.parquet"
        self.features_file = "features.parquet"
        self.train_file = "train.parquet"
        self.test_file = "test.parquet"
        self.default_user = "root"
        self.algorithm = ALGORITHM
        self.n_workers_cpu = N_WORKERS_CPU
        self.n_workers_gpu = N_WORKERS_GPU

        self.starting_time = time.time()

        # Define path
        self.path = Path(path)
        self.path_models = Path(path_models)
        self.db = self.path.joinpath(self.db_name)

        # create directories
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path.joinpath("static")).mkdir(parents=True, exist_ok=True)
        self.path_models.mkdir(exist_ok=True)

        # create or load a key to encrypt the tokens
        self.load_secret_key()

        # attributes of the server
        self.projects = {}
        self.db_manager = DatabaseManager(str(self.db))
        self.queue = Queue(
            nb_workers_cpu=self.n_workers_cpu,
            nb_workers_gpu=self.n_workers_gpu,
            path=self.path,
        )
        self.users = Users(self.db_manager)

        # update
        # TODO : manage better the closing
        self._running = True
        self._update_task = asyncio.create_task(self._update(timeout=UPDATE_TIMEOUT))

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
        self._running = False
        self.queue.executor.shutdown(wait=False)
        self.queue.close()
        print("Server off")

    async def _update(self, timeout: int = 1) -> None:
        """
        Update the queue with new tasks every X seconds
        """
        try:
            while self._running:
                print("update orchestrator")
                self.update()
                await asyncio.sleep(timeout)
        except asyncio.CancelledError:
            print("Update task cancelled.")
        finally:
            print("Update task finished.")

    def load_secret_key(self) -> None:
        """
        Load secret key in the environment
        - if key.yaml exists, load the key
        - if not, create a new key and the file
        """
        if (self.path.joinpath("key.yaml")).exists():
            with open(self.path.joinpath("key.yaml"), "r") as f:
                conf = yaml.safe_load(f)
            key = conf["key"]
        else:
            key = Fernet.generate_key().decode()
            with open(self.path.joinpath("key.yaml"), "w") as f:
                yaml.safe_dump({"key": key}, f)
        os.environ["SECRET_KEY"] = key

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

    def get_logs(
        self, project_slug: str, limit: int, partial: bool = True
    ) -> pd.DataFrame:
        """
        Get logs for a user/project
        project_slug: project slug or "all"
        """
        logs = self.db_manager.logs_service.get_logs("all", project_slug, limit)
        df = pd.DataFrame(
            logs, columns=["id", "time", "user", "project", "action", "NA"]
        )
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

    def get_server_state(self) -> ServerStateModel:
        # active projects
        active_projects = {}
        for p in self.projects:
            active_projects[p] = [
                {
                    "unique_id": c.unique_id,
                    "user": c.user,
                    "kind": c.kind,
                    "time": c.time,
                }
                for c in self.projects[p].computing
            ]

        # running processes
        q = self.queue.state()
        queue = {i: q[i] for i in q if q[i]["state"] in ["pending", "running"]}

        # server state
        gpu = get_gpu_memory_info()
        cpu = psutil.cpu_percent()
        cpu_count = psutil.cpu_count()
        memory_info = psutil.virtual_memory()
        disk_info = psutil.disk_usage("/")
        at_memory = get_dir_size(os.environ["ACTIVETIGGER_PATH"])

        return ServerStateModel(
            version=__version__,
            active_projects=active_projects,
            queue=queue,
            gpu=gpu,
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
        )

    def get_auth_projects(self, username: str) -> list[ProjectSummaryModel]:
        """
        Get projects authorized for the user
        """
        projects_auth = self.users.get_auth_projects(username)
        return [
            ProjectSummaryModel(
                user_right=i[1],
                parameters=ProjectModel(**i[2]),
                created_by=i[3],
                created_at=i[4].strftime("%Y-%m-%d %H:%M:%S"),
                size=round(
                    get_dir_size(os.environ["ACTIVETIGGER_PATH"] + "/" + i[0]), 1
                ),
                last_activity=self.db_manager.logs_service.get_last_activity_project(
                    i[0]
                ),
                project_slug=i[0],
            )
            for i in list(reversed(projects_auth))
        ]

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

    def create_access_token(self, data: dict, expires_min: int = 60):
        """
        Create access token
        """
        # create the token
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(
            to_encode, os.environ["SECRET_KEY"], algorithm=self.algorithm
        )

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
        payload = jwt.decode(
            token, os.environ["SECRET_KEY"], algorithms=[self.algorithm]
        )
        return payload

    def start_project(self, project_slug: str) -> dict:
        """
        Load project in server
        """
        if not self.exists(project_slug):
            raise Exception("This project does not exist")

        try:
            self.projects[project_slug] = Project(
                project_slug, self.queue, self.db_manager, path_models=self.path_models
            )
            return {"success": "Project loaded"}
        except Exception as e:
            raise Exception from e

    def set_project_parameters(self, project: ProjectModel, username: str) -> dict:
        """
        Update project parameters in the DB
        """

        # get project
        existing_project = self.db_manager.projects_service.get_project(
            project.project_slug
        )

        if existing_project:
            # Update the existing project
            self.db_manager.projects_service.update_project(
                project.project_slug, jsonable_encoder(project)
            )
            return {"success": "project updated"}
        else:
            # Insert a new project
            self.db_manager.projects_service.add_project(
                project.project_slug, jsonable_encoder(project), username
            )
            return {"success": "project added"}

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        existing_projects = self.db_manager.projects_service.existing_projects()
        return existing_projects

    def create_project(self, params: ProjectBaseModel, username: str) -> dict:
        """
        Set up a new project
        - load data and save
        - initialize parameters in the db
        - initialize files
        - add preliminary tags

        Comments:
        - when saved, the files followed the nomenclature of the project : text, label, etc.
        """

        print("Project", params)

        # test if possible to create the project
        if self.exists(params.project_name):
            raise Exception("This project already exists")

        # test if the name of the column is specified
        if params.col_id is None or params.col_id == "":
            raise Exception("No column selected for the id")
        if params.cols_text is None or len(params.cols_text) == 0:
            raise Exception("No column selected for the text")

        # get the slug of the project name as a key
        project_slug = slugify(params.project_name)

        # add the dedicated directory
        params.dir = self.path.joinpath(project_slug)

        # check if the directory already exists + file (should with the data)
        if not params.dir.exists():
            raise Exception("The directory does not exist and should")

        # Step 1 : load all data and index to str and rename columns
        if params.filename.endswith(".csv"):
            content = pd.read_csv(
                params.dir.joinpath(params.filename), low_memory=False
            )
        elif params.filename.endswith(".parquet"):
            content = pd.read_parquet(params.dir.joinpath(params.filename))
        elif params.filename.endswith(".xlsx"):
            content = pd.read_excel(params.dir.joinpath(params.filename))
        else:
            raise Exception("File format not supported (only csv, xlsx and parquet)")

        # rename columns both for data & params to avoid confusion
        content.columns = ["dataset_" + i for i in content.columns]  # type: ignore[assignment]
        if params.col_id:
            params.col_id = "dataset_" + params.col_id
        # change also the name in the parameters
        params.cols_text = ["dataset_" + i for i in params.cols_text if i]
        params.cols_context = ["dataset_" + i for i in params.cols_context if i]
        params.cols_label = ["dataset_" + i for i in params.cols_label if i]
        params.cols_stratify = ["dataset_" + i for i in params.cols_stratify if i]

        # remove completely empty lines
        content = content.dropna(how="all")
        all_columns = list(content.columns)
        n_total = len(content)

        # test if the size of the sample requested is possible
        if len(content) < params.n_test + params.n_train:
            shutil.rmtree(params.dir)
            raise Exception(
                f"Not enough data for creating the train/test dataset. Current : {len(content)} ; Selected : {params.n_test + params.n_train}"
            )

        # create the index
        keep_id = []  # keep unchanged the index to avoid desindexing

        # case of the index should be the row number
        if params.col_id == "dataset_row_number":
            print("Use the row number as index")
            content["id"] = [str(i) for i in range(len(content))]
            content.set_index("id", inplace=True)
        # case of a column as index
        else:
            # check if index after slugify is unique otherwise throw an error
            if not (
                (content[params.col_id].astype(str).apply(slugify)).nunique()
                == len(content)
            ):
                shutil.rmtree(params.dir)
                raise Exception(
                    "The id column is not unique after slugify, please change it"
                )
            content["id"] = content[params.col_id].astype(str).apply(slugify)
            keep_id.append(params.col_id)
            content.set_index("id", inplace=True)

        # convert columns that can be numeric or force text
        for col in content.columns:
            try:
                content[col] = pd.to_numeric(content[col], errors="raise")
            except Exception:
                content[col] = content[col].astype(str).replace("nan", None)

        # create the text column, merging the different columns
        content["text"] = content[params.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # convert NA texts in empty string
        content["text"] = content["text"].fillna("")

        # limit of usable text (in the futur, will be defined by the number of token)
        def limit(text):
            return 1200

        content["limit"] = content["text"].apply(limit)

        # save a complete copy of the dataset
        content.to_parquet(params.dir.joinpath(self.data_all), index=True)

        # ------------------------
        # End of the data cleaning
        # ------------------------

        # Step 2 : test dataset : from the complete dataset + random/stratification
        rows_test = []
        params.test = False
        testset = None
        if params.n_test != 0:
            # if no stratification
            if len(params.cols_stratify) == 0:
                testset = content.sample(params.n_test)
            # if stratification, total cat, number of element per cat, sample with a lim
            else:
                df_grouped = content.groupby(params.cols_stratify, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(params.n_test / nb_cat)
                testset = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat))
                )
            # save the testset
            testset.to_parquet(params.dir.joinpath(self.test_file), index=True)
            params.test = True
            rows_test = list(testset.index)

        # Step 3 : train dataset / different strategies

        # remove test rows
        content = content.drop(rows_test)

        # case where there is no test set and the selection is deterministic
        if not params.random_selection and params.n_test == 0:
            print("deterministic selection of the trainset")
            trainset = content[0 : params.n_train]
        # case to force the max of label from one column
        elif params.force_label and len(params.cols_label) > 0:
            print("force the selection of labels")
            f_notna = content[params.cols_label[0]].notna()
            f_na = content[params.cols_label[0]].isna()
            # different case regarding the number of labels
            if f_notna.sum() > params.n_train:
                trainset = content[f_notna].sample(params.n_train)
            else:
                n_train_random = params.n_train - f_notna.sum()
                trainset = pd.concat(
                    [content[f_notna], content[f_na].sample(n_train_random)]
                )
        # case there is stratification on the trainset
        elif len(params.cols_stratify) > 0 and params.stratify_train:
            print("stratification of the trainset")
            df_grouped = content.groupby(params.cols_stratify, group_keys=False)
            nb_cat = len(df_grouped)
            nb_elements_cat = round(params.n_train / nb_cat)
            trainset = df_grouped.apply(
                lambda x: x.sample(min(len(x), nb_elements_cat))
            )
        # default with random selection in the remaining elements
        else:
            print("random selection of the trainset")
            trainset = content.sample(params.n_train)

        # write the trainset
        trainset[
            list(set(["text", "limit"] + params.cols_context + keep_id))
        ].to_parquet(params.dir.joinpath(self.train_file), index=True)
        trainset[[]].to_parquet(params.dir.joinpath(self.features_file), index=True)

        # add an empty default scheme
        self.db_manager.projects_service.add_scheme(
            project_slug, "default", [], "multiclass", "system"
        )
        params.default_scheme = []

        # add loaded schemes from columns
        for col in params.cols_label:

            # select the type of scheme
            scheme_name = slugify(col).replace("dataset-", "")
            delimiters = content[col].str.contains("|", regex=False).sum()
            if delimiters < 5:
                scheme_type = "multiclass"
                scheme_labels = list(content[col].dropna().unique())
            else:
                scheme_type = "multilabel"
                scheme_labels = list(
                    content[col].dropna().str.split("|").explode().unique()
                )

            # check there is a limited number of labels
            if scheme_type == "multiclass" and len(scheme_labels) > 30:
                print("Too many different labels for multiclass > 30")
                continue

            print("Add scheme/labels from file in train/test")

            # add the scheme in the database
            self.db_manager.projects_service.add_scheme(
                project_slug,
                scheme_name,
                scheme_labels,
                scheme_type,
                "system",
            )

            # add the labels from the trainset in the database
            elements = [
                {"element_id": element_id, "annotation": label, "comment": ""}
                for element_id, label in trainset[col].dropna().items()
            ]
            self.db_manager.projects_service.add_annotations(
                dataset="train",
                user=username,
                project_slug=project_slug,
                scheme=scheme_name,
                elements=elements,
            )
            # add the labels from the trainset in the database if exists & not clear
            if isinstance(testset, pd.DataFrame) and not params.clear_test:
                elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in testset[col].dropna().items()
                ]
                self.db_manager.projects_service.add_annotations(
                    dataset="test",
                    user=username,
                    project_slug=project_slug,
                    scheme=scheme_name,
                    elements=elements,
                )

        # add user right on the project + root
        self.users.set_auth(username, project_slug, "manager")
        self.users.set_auth("root", project_slug, "manager")

        # save parameters (without the data)
        # params.cols_label = []  # reverse dummy
        project = params.model_dump()

        # add elements for the parameters
        project["project_slug"] = project_slug
        project["all_columns"] = all_columns
        project["n_total"] = n_total

        # save the parameters
        self.set_project_parameters(ProjectModel(**project), username)

        # delete the initial file
        params.dir.joinpath(params.filename).unlink()

        return {"success": project_slug}

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

    def clean_project(
        self, project_slug: str | None = None, project_name: str | None = None
    ) -> None:
        """
        Clean a project
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
        shutil.rmtree(self.path.joinpath(project_slug_verif), ignore_errors=True)

        ## remove static files
        if Path(
            f"{os.environ['ACTIVETIGGER_PATH']}/static/{project_slug_verif}"
        ).exists():
            shutil.rmtree(
                f"{os.environ['ACTIVETIGGER_PATH']}/static/{project_slug_verif}"
            )

    def update(self):
        """
        Update the state of the orchestrator
        - projects
        - state of the server
        """
        self.queue.clean_old_processes()
        timer = time.time()
        to_del = []
        for p, project in self.projects.items():
            # if project existing since one day, remove it from memory
            if (timer - project.starting_time) > 86400:
                to_del.append(p)
                continue
            # update the project
            project.update_processes()

        # remove the projects from memory
        for p in to_del:
            del self.projects[p]

        # update the information on the state of the project
        self.server_state = self.get_server_state()

    def create_dummy_project(self, username: str) -> None:
        """
        Create a dummy project for a user
        """

        # create name of the project and the directory
        project_name = "dummy_" + username
        project_slug = slugify(project_name)
        project_path = self.path.joinpath(project_slug)
        os.makedirs(project_path, exist_ok=True)

        # create and save a toy dataset
        newsgroups = fetch_20newsgroups(
            subset="all", remove=("headers", "footers", "quotes")
        )
        df = pd.DataFrame({"text": newsgroups.data, "target": newsgroups.target})
        df["category"] = df["target"].apply(lambda x: newsgroups.target_names[x])
        df.to_csv(project_path.joinpath("dummy.csv"), index=False)

        # parameters of the project
        params = ProjectBaseModel(
            project_name=project_name,
            language="en",
            filename="dummy.csv",
            col_id="row_number",
            cols_text=["text"],
            cols_label=["category"],
            n_train=3000,
            n_test=1000,
        )

        self.create_project(params, username)


# launch the instance
orchestrator = Orchestrator(
    os.environ.get("ACTIVETIGGER_PATH", "./projects"),
    os.environ.get("ACTIVETIGGER_MODEL", "./models"),
)
