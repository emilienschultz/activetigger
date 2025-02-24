"""
Define the orchestrator and launch the instance
"""

import logging
import os
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from cryptography.fernet import Fernet
from fastapi.encoders import jsonable_encoder
from jose import jwt
from slugify import slugify

from activetigger.datamodels import (
    ProjectDataModel,
    ProjectModel,
    ProjectSummaryModel,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.project import Project
from activetigger.queue import Queue
from activetigger.users import Users

logger = logging.getLogger("server")


# conf deployment
ALGORITHM = "HS256"
MAX_LOADED_PROJECTS = 20
N_WORKERS = 2


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
    n_workers: int
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
        """

        # unix system : set priority to this process
        try:
            os.nice(-15)
            print(f"Process niceness set to {os.nice(0)}")
        except PermissionError:
            print("You need administrative privileges to set negative niceness values.")

        self.max_projects = MAX_LOADED_PROJECTS
        self.db_name = "activetigger.db"
        self.data_all = "data_all.parquet"
        self.features_file = "features.parquet"
        self.train_file = "train.parquet"
        self.test_file = "test.parquet"
        self.default_user = "root"
        self.algorithm = ALGORITHM
        self.n_workers = N_WORKERS

        self.starting_time = time.time()

        # Define path
        self.path = Path(path)
        self.path_models = Path(path_models)

        # create or load a key
        self.load_secret_key()

        # if a YAML configuration file exists, overwrite
        if Path("config.yaml").exists():
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            if "path" in config:
                self.path = Path(config["path"])
            if "path_models" in config:
                self.path_models = Path(config["path_models"])

        self.db = self.path.joinpath(self.db_name)

        # create directories
        (self.path.joinpath("static")).mkdir(parents=True, exist_ok=True)
        self.path_models.mkdir(exist_ok=True)

        # attributes of the server
        self.projects = {}
        self.db_manager = DatabaseManager(str(self.db))
        self.queue = Queue(self.n_workers)
        self.users = Users(self.db_manager)

        # logging
        logging.basicConfig(
            filename=self.path.joinpath("log_server.log"),
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    def __del__(self):
        """
        Close the server
        """
        print("Ending the server")
        logger.error("Disconnect server")
        self.queue.executor.shutdown()
        self.queue.close()
        print("Server off")

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
        self.db_manager.projects_service.add_log(user, action, project, connect)
        logger.info("%s from %s in project %s", action, user, project)

    def get_logs(
        self, project_slug: str, limit: int, partial: bool = True
    ) -> pd.DataFrame:
        """
        Get logs for a user/project
        project_slug: project slug or "all"
        """
        logs = self.db_manager.projects_service.get_logs("all", project_slug, limit)
        df = pd.DataFrame(
            logs, columns=["id", "time", "user", "project", "action", "NA"]
        )
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

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

        self.projects[project_slug] = Project(
            project_slug, self.queue, self.db_manager, path_models=self.path_models
        )
        return {"success": "Project loaded"}

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

    def create_project(self, params: ProjectDataModel, username: str) -> dict:
        """
        Set up a new project
        - load data and save
        - initialize parameters in the db
        - initialize files
        - add preliminary tags

        Comments:
        - when saved, the files followed the nomenclature of the project : text, label, etc.
        """

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

        # create dedicated directory
        params.dir = self.path.joinpath(project_slug)
        if params.dir is None:
            raise Exception("Error in the creation of the directory")
        os.makedirs(params.dir)

        # copy total dataset as a copy (csv for the moment)
        with open(params.dir.joinpath("data_raw.csv"), "w") as f:
            f.write(params.csv)

        # Step 1 : load all data and index to str and rename columns
        content = pd.read_csv(params.dir.joinpath("data_raw.csv"), dtype=str)

        # rename columns both for data & params to avoid confusion
        content.columns = ["dataset_" + i for i in content.columns]
        params.col_id = "dataset_" + params.col_id if params.col_id else None
        params.cols_text = ["dataset_" + i for i in params.cols_text if i]
        params.cols_context = ["dataset_" + i for i in params.cols_context if i]
        params.col_label = "dataset_" + params.col_label if params.col_label else None
        params.cols_test = ["dataset_" + i for i in params.cols_test if i]

        # remove empty lines
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

        # create the text column, merging the different columns
        content["text"] = content[params.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # drop NA texts
        n_before = len(content)
        content.dropna(subset=["text"], inplace=True)
        if n_before != len(content):
            print(f"Drop {n_before - len(content)} empty text lines")

        # manage the label column
        if (params.col_label is not None) & (params.col_label != ""):
            content.rename(columns={params.col_label: "label"}, inplace=True)
        else:
            content["label"] = None
            params.col_label = None

        # limit of usable text (in the futur, will be defined by the number of token)
        def limit(text):
            return 1200

        content["limit"] = content["text"].apply(limit)

        # save a complete copy of the dataset
        content.to_parquet(params.dir.joinpath(self.data_all), index=True)

        # Step 2 : test dataset : from the complete dataset + random/stratification
        rows_test = []
        params.test = False
        testset = None
        if params.n_test != 0:
            # if no stratification
            if len(params.cols_test) == 0:
                testset = content.sample(params.n_test)
            # if stratification, total cat, number of element per cat, sample with a lim
            else:
                df_grouped = content.groupby(params.cols_test, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(params.n_test / nb_cat)
                testset = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat))
                )
            testset.to_parquet(params.dir.joinpath(self.test_file), index=True)
            params.test = True
            rows_test = list(testset.index)

        # Step 3 : train dataset, remove test rows, prioritize labelled data
        content = content.drop(rows_test)
        f_notna = content["label"].notna()
        f_na = content["label"].isna()

        # case where the order is kept and no testset
        if not params.random_selection and params.n_test == 0:
            trainset = content[0 : params.n_train]
        # case where there is more labelled data than needed
        elif f_notna.sum() > params.n_train:
            trainset = content[f_notna].sample(params.n_train)
        # case where there is less labelled data than needed so complete with random
        else:
            n_train_random = params.n_train - f_notna.sum()
            trainset = pd.concat(
                [content[f_notna], content[f_na].sample(n_train_random)]
            )

        trainset[list(set(["text"] + params.cols_context + keep_id))].to_parquet(
            params.dir.joinpath(self.train_file), index=True
        )
        trainset[[]].to_parquet(params.dir.joinpath(self.features_file), index=True)

        # if the case, add existing annotations in the database
        if params.col_label is None:
            self.db_manager.projects_service.add_scheme(
                project_slug, "default", [], "multiclass", "system"
            )
        else:
            # determine if multiclass / multilabel (arbitrary rule)
            delimiters = content["label"].str.contains("|", regex=False).sum()
            print("DELIMITERS", delimiters)

            # check there is a limited number of labels
            df = content["label"].dropna()
            params.default_scheme = list(df.unique())

            if len(params.default_scheme) < 30:
                print("Add scheme/labels from file in train/test")

                # add the scheme in the database
                self.db_manager.projects_service.add_scheme(
                    project_slug,
                    "default",
                    list(params.default_scheme),
                    "multiclass" if delimiters < 5 else "multilabel",
                    "system",
                )

                # add the labels from the trainset in the database
                elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in trainset["label"].dropna().items()
                ]
                self.db_manager.projects_service.add_annotations(
                    dataset="train",
                    user=username,
                    project_slug=project_slug,
                    scheme="default",
                    elements=elements,
                )
                # add the labels from the trainset in the database if exists & not clear
                if isinstance(testset, pd.DataFrame) and not params.clear_test:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in testset["label"].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="test",
                        user=username,
                        project_slug=project_slug,
                        scheme="default",
                        elements=elements,
                    )
            else:
                print("Too many different labels > 30")

        # add user right on the project + root
        self.users.set_auth(username, project_slug, "manager")
        self.users.set_auth("root", project_slug, "manager")

        # save parameters (without the data)
        params.col_label = None  # reverse dummy
        project = params.model_dump()

        # add elements for the parameters
        project["project_slug"] = project_slug
        project["all_columns"] = all_columns
        project["n_total"] = n_total

        # save the parameters
        self.set_project_parameters(ProjectModel(**project), username)

        # clean
        os.remove(params.dir.joinpath("data_raw.csv"))

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

    def update(self):
        """
        Update state of projects from the queue
        """
        timer = time.time()
        to_del = []
        self.queue.check()  # check if the queue is still up
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


# launch the instance
orchestrator = Orchestrator()
