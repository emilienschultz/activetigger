import concurrent.futures
import json
import logging
import os
import re
import secrets
import shutil
import time
import uuid
from datetime import datetime, timedelta, timezone
from multiprocessing import Manager
from multiprocessing.managers import SyncManager
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml
from fastapi.encoders import jsonable_encoder
from jose import jwt
from pandas import DataFrame, Series
from pydantic import ValidationError
from slugify import slugify

import activetigger.functions as functions
from activetigger.datamodels import (
    GenerateModel,
    ProjectDataModel,
    ProjectModel,
    ProjectSummaryModel,
    SimpleModelModel,
    TestSetDataModel,
    UserInDBModel,
)
from activetigger.db import DatabaseManager
from activetigger.models import BertModels, SimpleModels

logger = logging.getLogger("server")

db_name = "activetigger.db"
features_file = "features.parquet"
labels_file = "labels.parquet"
data_file = "data.parquet"
test_file = "test.parquet"
default_user = "root"
ALGORITHM = "HS256"


class Queue:
    """
    Managining parallel processes for computation
    Jobs with  concurrent.futures.ProcessPoolExecutor

    TODO : better management of failed processes
    """

    nb_workers: int
    executor: concurrent.futures.ProcessPoolExecutor
    manager: SyncManager
    current: dict

    def __init__(self, nb_workers: int = 2) -> None:
        """
        Initiating the queue
        """
        self.nb_workers = nb_workers
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.nb_workers
        )  # manage parallel processes
        self.manager = Manager()  # communicate within processes
        self.current = {}  # keep track of the current stack

        logger.info("Init Queue")

    def close(self) -> None:
        """
        Close the executor
        """
        self.executor.shutdown(cancel_futures=True, wait=False)
        self.manager.shutdown()
        logger.info("Close queue")
        print("Queue closes")

    # def check_failed_processes(self, future):
    #     """
    #     Check if a future failed.
    #     """
    #     try:
    #         result = future.result()  # Will raise if the task failed
    #         return False
    #     except Exception as e:
    #         return True

    def check(self) -> None:
        """
        Check if the exector still works, if not recreate it
        """
        try:
            self.executor.submit(lambda: None)
        except Exception:
            self.executor.shutdown(cancel_futures=True)
            self.executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=self.nb_workers
            )
            logger.error("Restart executor")
            print("Problem with executor ; restart")

    def add(self, kind: str, func: Callable, args: dict) -> str:
        """
        Add new element to queue
        """
        # generate a unique id
        unique_id = str(uuid.uuid4())

        # create an event to control the process
        event = self.manager.Event()
        args["event"] = event
        args["unique_id"] = unique_id

        # send the process to the executor
        try:
            future = self.executor.submit(func, **args)
        except Exception as e:
            logger.error(f"Error submitting task: {e}")
            return "error"

        # save in the stack
        self.current[unique_id] = {"kind": kind, "future": future, "event": event}
        return unique_id

    def kill(self, unique_id: str) -> dict:
        """
        Send a kill process with the event manager
        """
        if not unique_id in self.current:
            return {"error": "Id does not exist"}
        self.current[unique_id]["event"].set()
        self.delete(unique_id)
        return {"success": "Process killed"}

    def delete(self, ids: str | list) -> None:
        """
        Delete completed elements from the stack
        """
        if type(ids) is str:
            ids = [ids]
        for i in ids:
            if not self.current[i]["future"].done():
                print("Deleting a unfinished process")
            del self.current[i]

    def state(self) -> dict:
        """
        Return state of the queue
        """
        r = {}
        for f in self.current:
            if self.current[f]["future"].running():
                info = "running"
                exception = None
            else:
                info = "done"
                exception = self.current[f]["future"].exception()
            r[f] = {"state": info, "exception": exception}
        return r

    def get_nb_active_processes(self) -> int:
        """
        Number of active processes
        """
        return len([f for f in self.current if self.current[f]["future"].running()])


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
                if not user in existing:
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
        self.db_manager.add_auth(project_slug, username, status)
        return {"success": "Auth added to database"}

    def delete_auth(self, username: str, project_slug: str):
        """
        Delete user auth
        """
        self.db_manager.delete_auth(project_slug, username)
        return {"success": "Auth deleted"}

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

    def existing_users(self) -> list:
        """
        Get existing users
        (except root which can't be modified)
        """
        users = self.db_manager.get_users()
        return users

    def add_user(
        self, name: str, password: str, role: str = "manager", created_by: str = "NA"
    ) -> bool:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist
        if name in self.existing_users():
            return {"error": "Username already exists"}
        hash_pwd = functions.get_hash(password)
        self.db_manager.add_user(name, hash_pwd, role, created_by)

        return {"success": "User added to the database"}

    def delete_user(self, name: str) -> dict:
        """
        Deleting user
        """
        # specific cases
        if not name in self.existing_users():
            return {"error": "Username does not exist"}
        if name == "root":
            return {"error": "Can't delete root user"}

        # delete the user
        self.db_manager.delete_user(name)

        return {"success": "User deleted"}

    def get_user(self, name) -> UserInDBModel | dict:
        """
        Get user from database
        """
        if not name in self.existing_users():
            return {"error": "Username doesn't exist"}
        user = self.db_manager.get_user(name)
        return UserInDBModel(
            username=name, hashed_password=user["key"], status=user["description"]
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
        if not functions.compare_to_hash(password, user.hashed_password):
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


class Server:
    """
    Server to manage backend
    """

    db_name: str
    features_file: str
    labels_file: str
    data_file: str
    test_file: str
    default_user: str
    ALGORITHM: str
    n_workers: int = 2
    starting_time: float = None
    SECRET_KEY: str
    path: Path
    path_models: Path
    db: Path
    projects: dict
    db_manager: DatabaseManager
    queue: Queue
    users: Users

    def __init__(self, path=".", path_models="./models") -> None:
        """
        Start the server
        """
        self.db_name = db_name
        self.features_file = features_file
        self.labels_file = labels_file
        self.data_file = data_file
        self.test_file = test_file
        self.default_user = default_user
        self.ALGORITHM = ALGORITHM

        self.starting_time = time.time()
        self.SECRET_KEY = secrets.token_hex(32)

        # Define path
        self.path = Path(path)
        self.path_models = Path(path_models)
        # if a YAML configuration file exists, overwrite
        if Path("config.yaml").exists():
            with open("config.yaml") as f:
                config = yaml.safe_load(f)
            if "path" in config:
                self.path = Path(config["path"])
            if "path_models" in config:
                self.path_models = Path(config["path_models"])

        self.db = self.path / self.db_name

        # create directories
        if not self.path.exists():
            os.makedirs(self.path)
        if not (self.path / "static").exists():
            os.mkdir((self.path / "static"))
        if not self.path_models.exists():
            os.makedirs(self.path_models)

        # attributes of the server
        self.projects: dict = {}
        self.db_manager = DatabaseManager(self.db)
        self.queue = Queue(self.n_workers)
        self.users = Users(self.db_manager)

    def __del__(self):
        """
        Close the server
        """
        print("Ending the server")
        logger.error("Disconnect server")
        self.queue.executor.shutdown()
        self.queue.close()
        print("Server off")

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
        self.db_manager.add_log(user, action, project, connect)
        logger.info(f"{action} from {user} in project {project}")

    def get_logs(self, username: str, project_slug: str, limit: int) -> pd.DataFrame:
        """
        Get logs for a user/project
        """
        logs = self.db_manager.get_logs(username, project_slug, limit)
        df = pd.DataFrame(
            logs, columns=["id", "time", "user", "project", "action", "NA"]
        )
        return df

    def get_projects(self, username: str) -> dict[dict]:
        """
        Get projects authorized for the user
        """

        projects_auth = self.users.get_auth_projects(username)
        return [
            ProjectSummaryModel(
                user_right=i[1],
                parameters=ProjectModel(**json.loads(i[2])),
                created_by=i[3],
                created_at=i[4].strftime("%Y-%m-%d %H:%M:%S"),
            )
            for i in projects_auth
        ]

    def get_project_params(self, project_slug: str) -> ProjectModel | None:
        """
        Get project from database
        """
        existing_project = self.db_manager.get_project(project_slug)
        if existing_project:
            return ProjectModel(**json.loads(existing_project["parameters"]))
        else:
            return None

    def exists(self, project_name: str) -> bool:
        """
        Test if a project exists in the database
        with a sluggified form (to be able to use it in URL)
        """
        return slugify(project_name) in self.existing_projects()

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        existing_projects = self.db_manager.existing_projects()
        return existing_projects

    def create_access_token(self, data: dict, expires_min: int = 60):
        """
        Create access token
        """
        # create the token
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.SECRET_KEY, algorithm=self.ALGORITHM)

        # add it in the database as active
        self.db_manager.add_token(encoded_jwt, "active")

        # return it
        return encoded_jwt

    def revoke_access_token(self, token) -> None:
        """
        Revoke existing access token
        """
        self.db_manager.revoke_token(token)
        return None

    def decode_access_token(self, token: str):
        """
        Decode access token
        """
        # get status
        status = self.db_manager.get_token_status(token)
        if status != "active":
            return {"error": "Token not valid"}

        # decode payload
        payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
        return payload

    def start_project(self, project_slug: str) -> dict:
        """
        Load project in server
        """
        if not self.exists(project_slug):
            return {"error": "Project does not exist"}

        self.projects[project_slug] = Project(project_slug, self.queue, self.db_manager)
        return {"success": "Project loaded"}

    def set_project_parameters(self, project: ProjectModel, username: str) -> dict:
        """
        Update project parameters in the DB
        """

        # get project
        existing_project = self.db_manager.get_project(project.project_slug)

        if existing_project:
            # Update the existing project
            self.db_manager.update_project(
                project.project_slug, jsonable_encoder(project)
            )
            return {"success": "project updated"}
        else:
            # Insert a new project
            self.db_manager.add_project(
                project.project_slug, jsonable_encoder(project), username
            )
            return {"success": "project added"}

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
            return {"error": "Project name already exist"}

        # get the slug of the project name as a key
        project_slug = slugify(params.project_name)

        # create dedicated directory
        params.dir = self.path / project_slug
        if params.dir.exists():
            return {"error": "This name is already used"}
        os.makedirs(params.dir)

        # copy total dataset as a copy (csv for the moment)
        with open(params.dir / "data_raw.csv", "w") as f:
            f.write(params.csv)

        # TODO : maximise the aleardy tagged in the annotate dataset, and None in the test
        # if possible, annotated data in the annotation dataset
        # if possible, test data without annotation
        # if n_test = 0, no test set
        # stratified if possible by cols_test

        # Step 1 : load all data and index to str and rename
        content = pd.read_csv(params.dir / "data_raw.csv")

        # quick fix to avoid problem with parquet index
        content = content.drop(
            columns=[i for i in content.columns if "__index_level" in i]
        )

        if len(content) < params.n_test + params.n_train:
            return {
                "error": f"Not enought data for creating the train/test dataset. Current : {len(content)} ; Selected : {params.n_test + params.n_train}"
            }

        # check if index is unique otherwise FORCE the index from 0 to N
        if not content[params.col_id].nunique() == len(content):
            print("There are duplicate in the column selected for index")
            content["id"] = range(0, len(content))
            params.col_id = "id"

        # rename columns that are going to be used & remove NA texts
        content = (
            content.rename(columns={params.col_id: "id", params.col_text: "text"})
            .set_index("id")  # set id as index
            .dropna(subset=["text"])
        )
        if params.col_label:
            content.rename(columns={params.col_label: "label"}, inplace=True)
        else:
            content["label"] = None

        # drop duplicated index and assure it is string
        content = content[~content.index.duplicated(keep="first")]
        content.index = [str(i) for i in list(content.index)]  # sure to be str

        # Information of the limit of usable text (in the futur, will be defined by the number of token)
        # but it depends of the tokenizer
        def limit(text):
            return 1200

        content["limit"] = content["text"].apply(limit)

        # Step 2 : test dataset, no already labelled data, random + stratification
        rows_test = []
        params.test = False
        if params.n_test != 0:
            # only on non labelled data
            f = content["label"].isna()
            if (f.sum()) < params.n_test:
                return {"error": "Not enought data for creating the test dataset"}
            if len(params.cols_test) == 0:  # if no stratification
                testset = content[f].sample(params.n_test)
            else:  # if stratification, total cat, number of element per cat, sample with a lim
                df_grouped = content[f].groupby(params.cols_test, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(params.n_test / nb_cat)
                testset = df_grouped.apply(
                    lambda x: x.sample(min(len(x), nb_elements_cat))
                )
            testset.to_parquet(params.dir / self.test_file, index=True)
            params.test = True
            rows_test = list(testset.index)

        # Step 3 : train dataset, remove test rows, prioritize labelled data
        content = content.drop(rows_test)
        f_notna = content["label"].notna()
        f_na = content["label"].isna()
        if (
            f_notna.sum() > params.n_train
        ):  # case where there is more labelled data than needed
            trainset = content[f_notna].sample(params.n_train)
        else:
            n_train_random = params.n_train - f_notna.sum()  # number of element to pick
            trainset = pd.concat(
                [content[f_notna], content[f_na].sample(n_train_random)]
            )

        trainset.to_parquet(params.dir / self.data_file, index=True)
        trainset[["text"] + params.cols_context].to_parquet(
            params.dir / self.labels_file, index=True
        )
        trainset[[]].to_parquet(params.dir / self.features_file, index=True)

        # if the case, add labels in the database
        if (not params.col_label is None) and ("label" in trainset.columns):
            print("Add scheme/labels from file")

            df = trainset["label"].dropna()
            params.default_scheme = list(df.unique())

            # add the scheme in the database
            self.db_manager.add_scheme(
                project_slug, "default", json.dumps(params.default_scheme), "file"
            )

            # add the labels in the database
            for element_id, label in df.items():
                self.db_manager.add_annotation(
                    action="add",
                    user=username,
                    project_slug=project_slug,
                    element_id=element_id,
                    scheme="default",
                    annotation=label,
                )
                print("add annotations ", element_id)

        # add user right on the project + root
        self.users.set_auth(username, project_slug, "manager")
        self.users.set_auth("root", project_slug, "manager")

        # save parameters (without the data)
        params.col_label = None  # reverse dummy
        project = params.model_dump()
        project["project_slug"] = project_slug
        self.set_project_parameters(ProjectModel(**project), username)

        return {"success": "Project created"}

    def delete_project(self, project_slug: str) -> dict:
        """
        Delete a project
        """

        if not self.exists(project_slug):
            return {"error": "Project doesn't exist"}

        # remove directory
        params = self.get_project_params(project_slug)
        shutil.rmtree(params.dir)

        # clean database
        self.db_manager.delete_project(project_slug)
        return {"success": "Project deleted"}


class Features:
    """
    Manage project features
    Comment :
    - as a file
    - use "__" as separator
    """

    project_slug: str
    path: Path
    queue: Queue
    informations: dict
    content: DataFrame
    map: dict
    training: dict
    projections: dict
    possible_projections: dict
    options: dict

    def __init__(self, project_slug: str, data_path: Path, queue) -> None:
        """
        Initit features
        """
        self.project_slug = project_slug
        self.path = data_path
        self.queue = queue
        self.informations = {}
        content, map = self.load()
        self.content = content
        self.map = map
        self.training: dict = {}

        # managing projections
        self.projections: dict = {}
        self.possible_projections: dict = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": "euclidean",
            },
            "tsne": {
                "n_components": 2,
                "learning_rate": "auto",
                "init": "random",
                "perplexity": 3,
            },
        }

        # options
        self.options: dict = {
            "sbert": {},
            "fasttext": {},
            "dfm": {
                "tfidf": False,
                "ngrams": 1,
                "min_term_freq": 5,
                "max_term_freq": 100,
                "norm": None,
                "log": None,
            },
            "regex": {"formula": None},
        }

    def __repr__(self) -> str:
        return f"Available features : {self.map}"

    def load(self):
        """
        Load file and agregate columns
        """

        def find_strings_with_pattern(strings, pattern):
            matching_strings = [s for s in strings if re.match(pattern, s)]
            return matching_strings

        data = pd.read_parquet(self.path)
        var = set([i.split("__")[0] for i in data.columns])
        dic = {i: find_strings_with_pattern(data.columns, i) for i in var}
        return data, dic

    def add(self, name: str, content: DataFrame | Series) -> dict:
        """
        Add feature(s) and save
        """
        # test length
        if len(content) != len(self.content):
            raise ValueError("Features don't have the right shape")

        if name in self.map:
            return {"error": "feature name already exists"}

        # change type
        if type(content) == Series:
            content = pd.DataFrame(content)

        # add to the table & dictionnary
        content.columns = [f"{name}__{i}" for i in content.columns]
        self.map[name] = list(content.columns)

        self.content = pd.concat([self.content, content], axis=1)
        # save
        self.content.to_parquet(self.path)

        return {"success": "feature added"}

    def delete(self, name: str):
        """
        Delete feature
        """
        if not name in self.map:
            return {"error": "feature doesn't exist"}

        col = self.get([name])
        del self.map[name]
        self.content = self.content.drop(columns=col)
        self.content.to_parquet(self.path)
        return {"success": "feature deleted"}

    def get(self, features: list | str = "all"):
        """
        Get content for specific features
        """
        if features == "all":
            features = list(self.map.keys())
        if type(features) is str:
            features = [features]

        cols = []
        missing = []
        for i in features:
            if i in self.map:
                cols += self.map[i]
            else:
                missing.append(i)

        if len(i) > 0:
            print("Missing features:", missing)
        return self.content[cols]

    def update_processes(self):
        """
        Check for computing processing completed
        and clean them for the queue
        """
        # for features
        for name in self.training.copy():
            unique_id = self.training[name]
            # case the process have been canceled, clean
            if unique_id not in self.queue.current:
                del self.training[name]
                continue
            # else check its state
            if self.queue.current[unique_id]["future"].done():
                r = self.queue.current[unique_id]["future"].result()
                if "error" in r:
                    print("Error in the feature processing", unique_id)
                else:
                    df = r["success"]
                    self.add(name, df)
                    self.queue.delete(unique_id)
                    del self.training[name]
                    print("Add feature", name)

        # for projections
        training = [u for u in self.projections if "queue" in self.projections[u]]
        for u in training:
            unique_id = self.projections[u]["queue"]
            if self.queue.current[unique_id]["future"].done():
                df = self.queue.current[unique_id]["future"].result()
                self.projections[u]["data"] = df
                self.projections[u]["id"] = self.projections[u]["queue"]
                del self.projections[u]["queue"]
                self.queue.delete(unique_id)

    def get_info(self):
        """
        Informations on features + update
        Comments:
            Maybe not the best solution
            Database ? How to avoid a loop ...
        """
        # update if new elements added in features
        for f in self.map:
            if ("regex_" in f) and (not f in self.informations):
                df = self.get(f)
                if len(df.columns) > 0:
                    self.informations[f] = int(df[df.columns[0]].sum())
        return dict(self.informations)


class Schemes:
    """
    Manage project schemes & tags

    Tables :
    - schemes
    - annotations
    """

    project_slug: str
    db_manager: DatabaseManager
    content: DataFrame
    test: DataFrame | None

    def __init__(
        self,
        project_slug: str,
        path_content: Path,  # training data
        path_test: Path,  # test data
        db_manager: DatabaseManager,
    ) -> None:
        """
        Init empty
        """
        self.project_slug = project_slug
        self.db_manager = db_manager
        self.content = pd.read_parquet(path_content)  # text + context
        self.test = None
        if path_test.exists():
            self.test = pd.read_parquet(path_test)

        available = self.available()

        # create a default scheme if not available
        if len(available) == 0:
            self.add_scheme(name="default", labels=[])

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available()}"

    def get_scheme_data(
        self, scheme: str, complete: bool = False, kind: list | str = ["add"]
    ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        Join with text data in separate file (train or test)

        Comments:
            For the moment tags can be add, test, predict, reconciliation

        TODO : replace all "add" with "train" in the code
        """
        if not scheme in self.available():
            return {"error": "Scheme doesn't exist"}

        if isinstance(kind, str):
            kind = [kind]

        # get all elements from the db
        # - last element for each id
        # - for a specific scheme

        results = self.db_manager.get_scheme_elements(self.project_slug, scheme, kind)

        # conn = sqlite3.connect(self.db)
        # cursor = conn.cursor()
        # action = "(" + " OR ".join([f"action = ?" for i in kind]) + ")"

        # query = f"""
        #     SELECT element_id, tag, user, MAX(time)
        #     FROM annotations
        #     WHERE scheme = ? AND project = ? AND {action}
        #     GROUP BY element_id
        #     ORDER BY time DESC;
        # """
        # cursor.execute(query, (scheme, self.project_slug) + tuple(kind))

        # results = cursor.fetchall()
        # conn.close()

        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "timestamp"]
        ).set_index("id")
        df.index = [str(i) for i in df.index]
        if complete:  # all the elements
            if "test" in kind:
                # case if the test, join the text data
                t = self.test[["text"]].join(df)
                return t
            else:
                return self.content.join(df)
        return df

    def get_reconciliation_table(self, scheme: str):
        """
        Get reconciliation table
        TODO : add the filter on action
        """
        if not scheme in self.available():
            return {"error": "Scheme doesn't exist"}

        # get the last tag for each id and each user
        # conn = sqlite3.connect(self.db)
        # cursor = conn.cursor()

        # query = f"""
        #     SELECT e.element_id, e.tag, e.user, e.time
        #     FROM annotations AS e
        #     INNER JOIN (
        #         SELECT id, user, MAX(time) AS last_timestamp
        #         FROM annotations
        #         WHERE project = ? AND scheme = ?
        #         GROUP BY element_id, user
        #     ) AS last_entries
        #     ON e.id = last_entries.id;
        # """
        # cursor.execute(query, (self.project_slug, scheme))
        # results = cursor.fetchall()
        # conn.close()

        results = self.db_manager.get_table_annotations_users(self.project_slug, scheme)
        # Shape the data
        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "time"]
        )  # shape as a dataframe
        agg = lambda x: list(x)[0] if len(x) > 0 else None  # take the label else None
        df = df.pivot_table(
            index="id", columns="user", values="labels", aggfunc=agg
        )  # pivot and keep the label
        f_multi = df.apply(
            lambda x: len(set([i for i in x if pd.notna(i)])) > 1, axis=1
        )  # filter for disagreement
        users = list(df.columns)
        df = pd.DataFrame(
            df.apply(lambda x: x.to_dict(), axis=1), columns=["annotations"]
        )
        df = df.join(self.content[["text"]], how="left")  # add the text
        df = df[f_multi].reset_index()
        # return the result
        return df, users

    def convert_tags(
        self, former_label: str, new_label: str, scheme: str, username: str
    ):
        """
        Convert tags from a specific label to another
        """
        # get id with the current tag
        df = self.get_scheme_data(scheme)
        to_recode = df[df["labels"] == former_label].index
        # for each of them, push the new tag
        for i in to_recode:
            self.push_tag(i, new_label, scheme, username, "add")
        return {"success": "All tags recoded"}

    def get_total(self, dataset="train"):
        """
        Number of element in the dataset
        """
        if dataset == "test":
            return len(self.test)
        return len(self.content)

    def get_table(
        self,
        scheme: str,
        min: int,
        max: int,
        mode: str,
        contains: str | None = None,
        set: str = "train",
        user: str = "all",
    ) -> DataFrame:
        """
        Get data table
        scheme : the annotations
        min, max: the range
        mode: sample
        contains: search
        user: select by user
        set: train or test

        Choice to order by index.
        """
        # check for errors
        if not mode in ["tagged", "untagged", "all", "recent"]:
            mode = "all"
        if not scheme in self.available():
            return {"error": "scheme not available"}

        # case of the test set, no fancy stuff
        if set == "test":
            print("test mode")
            df = self.get_scheme_data(scheme, complete=True, kind="test")
            print("df", df.shape)

            # normalize size
            if max == 0:
                max = len(df)
            if max > len(df):
                max = len(df)

            if min > len(df):
                return {"error": "min value too high"}

            return df.sort_index().iloc[min:max].reset_index()

        df = self.get_scheme_data(scheme, complete=True)

        # case of recent annotations (no filter possible)
        if mode == "recent":
            list_ids = self.db_manager.get_recent_annotations(
                self.project_slug, user, scheme, max - min
            )
            return df.loc[list_ids]

        # filter for contains
        if contains:
            f_contains = df["text"].str.contains(contains)
            df = df[f_contains]

        # build dataset
        if mode == "tagged":
            df = df[df["labels"].notnull()]
        if mode == "untagged":
            df = df[df["labels"].isnull()]

        # normalize size
        if max == 0:
            max = len(df)
        if max > len(df):
            max = len(df)

        if min > len(df):
            return {"error": "min value too high"}

        return df.sort_index().iloc[min:max].reset_index()

    def add_scheme(self, name: str, labels: list):
        """
        Add new scheme
        """
        if self.exists(name):
            return {"error": "scheme name already exists"}

        self.db_manager.add_scheme(self.project_slug, name, json.dumps(labels), None)

        return {"success": "scheme created"}

    def add_label(self, label: str, scheme: str, user: str):
        """
        Add label in a scheme
        """
        available = self.available()
        print("AVAILABLE", available)

        if (label is None) or (label == ""):
            return {"error": "the name is void"}
        if not scheme in available:
            return {"error": "scheme doesn't exist"}
        if available[scheme] is None:
            available[scheme] = []
        if label in available[scheme]:
            return {"error": "label already exist"}
        labels = available[scheme]
        labels.append(label)
        self.update_scheme(scheme, labels)
        return {"success": "scheme updated with a new label"}

    def exists_label(self, scheme: str, label: str):
        """
        Test if a label exist in a scheme
        """
        available = self.available()
        if not scheme in available:
            return {"error": "scheme doesn't exist"}
        if label in available[scheme]:
            return True
        return False

    def delete_label(self, label: str, scheme: str, user: str):
        """
        Delete a label in a scheme
        """
        available = self.available()
        if not scheme in available:
            return {"error": "scheme doesn't exist"}
        if not label in available[scheme]:
            return {"error": "label does not exist"}
        labels = available[scheme]
        labels.remove(label)
        # push empty entry for tagged elements
        df = self.get_scheme_data(scheme)
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            print(i)
            self.push_tag(i, None, scheme, user, "add")
        self.update_scheme(scheme, labels)
        return {"success": "scheme updated removing a label"}

    def update_scheme(self, scheme: str, labels: list):
        """
        Update existing schemes from database
        """
        self.db_manager.update_scheme(self.project_slug, scheme, json.dumps(labels))
        return {"success": "scheme updated"}

    def delete_scheme(self, name) -> dict:
        """
        Delete a scheme
        """
        self.db_manager.delete_scheme(self.project_slug, name)
        return {"success": "scheme deleted"}

    def exists(self, name: str) -> bool:
        """
        Test if scheme exist
        """
        if name in self.available():
            return True
        return False

    def available(self) -> dict:
        """
        Available schemes {scheme:[labels]}
        """
        r = self.db_manager.available_schemes(self.project_slug)
        return {i[0]: json.loads(i[1]) for i in r}

    def get(self) -> dict:
        """
        state of the schemes
        """
        r = {"project_slug": self.project_slug, "availables": self.available()}
        return r

    def delete_tag(self, element_id: str, scheme: str, user: str = "server") -> bool:
        """
        Delete a recorded tag
        i.e. : add empty label
        """

        self.db_manager.post_annotation(
            self.project_slug, scheme, element_id, None, user, "delete"
        )
        self.db_manager.post_annotation(
            self.project_slug, scheme, element_id, None, user, "add"
        )
        return True

    def push_tag(
        self,
        element_id: str,
        tag: str | None,
        scheme: str,
        user: str = "server",
        mode: str = "train",
    ):
        """
        Record a tag in the database
        mode : train, predict, test
        """

        # TO FIX IN THE FUTURE
        if mode == "train":
            mode = "add"

        # test if the action is possible
        a = self.available()
        if not scheme in a:
            return {"error": "scheme unavailable"}
        if (not tag is None) and (not tag in a[scheme]):
            return {"error": "this tag doesn't belong to this scheme"}

        # TODO : add a test also for testing
        # if (not element_id in self.content.index):
        #    return {"error":"element doesn't exist"}

        self.db_manager.post_annotation(
            self.project_slug, scheme, element_id, tag, user, mode
        )
        print(("push tag", mode, user, self.project_slug, element_id, scheme, tag))
        return {"success": "tag added"}

    def push_table(self, table, user: str, action: str = "add") -> bool:
        """
        Push table index/tags to update
        Comments:
        - only update modified labels
        """
        data = {i: j for i, j in zip(table.list_ids, table.list_labels)}
        for i in data:
            r = self.push_tag(i, data[i], table.scheme, user, action)
            if "error" in r:
                return {"error": "Something happened when recording."}
        return {"success": "table pushed"}

    def get_coding_users(self, scheme: str):
        """
        Get users action for a scheme
        """
        results = self.db_manager.get_coding_users(scheme, self.project_slug)
        return results


class Generations:
    """
    Class to manage generation data
    """

    queue: Queue
    generating: dict
    db_manager: DatabaseManager

    def __init__(self, queue: Queue, db_manager: DatabaseManager) -> None:
        self.queue = queue
        self.db_manager = db_manager
        self.generating = {}

    def update_generations(self):
        """
        Manage the process launched by the class
        Only one possible by user
        """
        for name in self.generating.copy():
            unique_id = self.generating[name]["unique_id"]

            # case the process have been canceled, clean
            if unique_id not in self.queue.current:
                del self.generating[name]
                continue

            # else check its state
            if self.queue.current[unique_id]["future"].done():
                r = self.queue.current[unique_id]["future"].result()
                if "error" in r:
                    print("Error in the generating process", unique_id)
                else:
                    results = r["success"]
                    for row in results:
                        self.add(
                            user=row["user"],
                            project_slug=row["project_slug"],
                            element_id=row["element_id"],
                            endpoint=row["endpoint"],
                            prompt=row["prompt"],
                            answer=row["answer"],
                        )
                    self.queue.delete(unique_id)
                    del self.generating[name]

    def add(
        self,
        user: str,
        project_slug: str,
        element_id: str,
        endpoint: str,
        prompt: str,
        answer: str,
    ) -> None:
        """
        Add a generated element in the database
        """
        self.db_manager.add_generated(
            user=user,
            project_slug=project_slug,
            element_id=element_id,
            endpoint=endpoint,
            prompt=prompt,
            answer=answer,
        )
        return None

    def get_generated(
        self,
        project_slug: str,
        username: str,
        n_elements: str,
    ) -> DataFrame:
        """
        Get generated elements from the database
        """
        result = self.db_manager.get_generated(
            project_slug=project_slug, username=username, n_elements=n_elements
        )
        df = pd.DataFrame(
            result, columns=["time", "index", "prompt", "answer", "endpoint"]
        )
        df["time"] = pd.to_datetime(df["time"])
        df["time"] = df["time"].dt.tz_localize("UTC")
        df["time"] = df["time"].dt.tz_convert("Europe/Paris")
        return df


class Project(Server):
    """
    Project object
    """

    starting_time: float
    name: str
    queue: Queue
    db_manager: DatabaseManager
    params: ProjectModel
    content: DataFrame
    schemes: Schemes
    features: Features
    bertmodels: BertModels
    simplemodels: SimpleModels
    generations: Generations

    def __init__(
        self,
        project_slug: str,
        queue: Queue,
        db_manager: DatabaseManager,
    ) -> None:
        """
        Load existing project
        """
        self.starting_time = time.time()
        self.name = project_slug
        self.queue = queue
        self.db_manager = db_manager
        self.params = self.load_params(project_slug)

        # check if directory exists
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # loading data
        self.content = pd.read_parquet(self.params.dir / data_file)

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.params.dir / labels_file,
            self.params.dir / test_file,
            self.db_manager,
        )
        self.features = Features(
            project_slug, self.params.dir / features_file, self.queue
        )
        self.bertmodels = BertModels(self.params.dir, self.queue)
        self.simplemodels = SimpleModels(self.params.dir, self.queue)
        self.generations = Generations(self.queue, self.db_manager)

    def __del__(self):
        pass

    def load_params(self, project_slug: str) -> ProjectModel:
        """
        Load params from database
        """
        existing_project = self.db_manager.get_project(project_slug)
        if existing_project:
            return ProjectModel(**json.loads(existing_project["parameters"]))
        else:
            raise NameError(f"{project_slug} does not exist.")

    def add_testdata(self, testset: TestSetDataModel):
        """
        Add a test dataset
        """
        if self.schemes.test is not None:
            return {"error": "Already a test dataset"}

        # write the buffer send by the frontend
        with open(self.params.dir / "test_set_raw.csv", "w") as f:
            f.write(testset.csv)

        # load it
        df = pd.read_csv(
            self.params.dir / "test_set_raw.csv",
            dtype={testset.col_id: str, testset.col_text: str},
            nrows=testset.n_test,
        )

        # change names
        df = df.rename(
            columns={testset.col_id: "id", testset.col_text: "text"}
        ).set_index("id")

        # write the dataset
        df[[testset.col_text]].to_parquet(self.params.dir / self.test_file)
        # load the data
        self.schemes.test = df[[testset.col_text]]
        # update parameters
        self.params.test = True

        return {"success": "test dataset added"}

    def update_simplemodel(
        self, simplemodel: SimpleModelModel, username: str, n_min: int = 10
    ) -> dict:
        """
        Update simplemodel on the base of an already existing
        simplemodel object

        n_min: minimal number of elements annotated
        """
        if simplemodel.features is None or len(simplemodel.features) == 0:
            return {"error": "Empty features"}
        if simplemodel.model not in list(self.simplemodels.available_models.keys()):
            return {"error": "Model doesn't exist"}
        if simplemodel.scheme not in self.schemes.available():
            return {"error": "Scheme doesn't exist"}
        if len(self.schemes.available()[simplemodel.scheme]) < 2:
            return {"error": "2 different labels needed"}

        # force dfm for multi_naivebayes
        if simplemodel.model == "multi_naivebayes":
            simplemodel.features = ["dfm"]
            simplemodel.standardize = False

        # test if the parameters have the correct format
        try:
            validation = self.simplemodels.validation[simplemodel.model]
            r = validation(**simplemodel.params)
        except ValidationError as e:
            return {"error": e.json()}

        df_features = self.features.get(simplemodel.features)
        df_scheme = self.schemes.get_scheme_data(scheme=simplemodel.scheme)

        # test for a minimum of annotated elements
        if len(df_scheme) < n_min:
            return {"error": f"there are less than {n_min} annotated rows"}

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)
        self.simplemodels.add_simplemodel(
            user=username,
            scheme=simplemodel.scheme,
            features=simplemodel.features,
            name=simplemodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=simplemodel.params,
            standardize=simplemodel.standardize,
        )

        return {"success": "Simplemodel updated"}

    def get_next(
        self,
        scheme: str,
        selection: str = "deterministic",
        sample: str = "untagged",
        user: str = "user",
        tag: None | str = None,
        history: list = [],
        frame: None | list = None,
        filter: str | None = None,
    ) -> dict:
        """
        Get next item for a specific scheme with a specific method
        - deterministic
        - random
        - active
        - maxprob
        - test

        filter is a regex to use on the corpus
        """

        if scheme not in self.schemes.available():
            return {"error": "Scheme doesn't exist"}

        # specific case of test, random element
        if selection == "test":
            df = self.schemes.get_scheme_data(scheme, complete=True, kind=["test"])
            f = df["labels"].isnull()
            element_id = df[f].sample(random_state=42).index[0]
            element = {
                "element_id": str(element_id),
                "text": df.loc[element_id, "text"],
                "selection": "test",
                "context": {},
                "info": "",
                "predict": {"label": None, "proba": None},
                "frame": [],
                "limit": 1200,
                "history": [],
            }
            return element

        # select the current state of annotation
        df = self.schemes.get_scheme_data(scheme, complete=True)

        # build filters regarding the selection mode
        f = df["labels"].apply(lambda x: True)
        if sample == "untagged":
            f = df["labels"].isnull()
        if sample == "tagged":
            f = df["labels"].notnull()

        # add a regex condition to the selection
        if filter:
            if "CONTEXT=" in filter:  # case to search in the context
                f_regex = (
                    df[self.params.cols_context]
                    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
                    .str.contains(
                        filter.replace("CONTEXT=", ""), regex=True, case=True, na=False
                    )
                )
            else:
                f_regex = df["text"].str.contains(
                    filter, regex=True, case=True, na=False
                )
            f = f & f_regex

        # manage frame selection (if projection, only in the box)
        if frame and len(frame) == 4:
            if user in self.features.projections:
                if "data" in self.features.projections[user]:
                    projection = self.features.projections[user]["data"]
                    f_frame = (
                        (projection[0] > frame[0])
                        & (projection[0] < frame[1])
                        & (projection[1] > frame[2])
                        & (projection[1] < frame[3])
                    )
                    f = f & f_frame
                else:
                    return {"error": "Data projection doesn't exist for this user"}
            else:
                return {"error": "Projection model doesn't exist for this user"}

        # test if there is at least one element available
        if sum(f) == 0:
            return {"error": "No element available with this selection mode."}

        # select type of selection
        ss = df[f].drop(history, errors="ignore")
        if len(ss) == 0:
            return {"error": "No element available with this selection mode."}
        indicator = None

        if selection == "deterministic":  # next row
            element_id = ss.index[0]

        if selection == "random":  # random row
            element_id = ss.sample(frac=1, random_state=42).index[0]

        # higher prob, only possible if the model has been trained
        if selection == "maxprob":
            if not self.simplemodels.exists(user, scheme):
                return {"error": "Simplemodel doesn't exist"}
            if tag is None:  # default label to first
                return {"error": "Select a tag"}
            sm = self.simplemodels.get_model(user, scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            element_id = (
                proba[f][tag]
                .drop(history, errors="ignore")
                .sort_values(ascending=False)
                .index[0]
            )  # get max proba id
            indicator = f"probability: {round(proba.loc[element_id,tag],2)}"

        # higher entropy, only possible if the model has been trained
        if selection == "active":
            if not self.simplemodels.exists(user, scheme):
                return {"error": "Simplemodel doesn't exist"}
            sm = self.simplemodels.get_model(user, scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            element_id = (
                proba[f]["entropy"]
                .drop(history, errors="ignore")
                .sort_values(ascending=False)
                .index[0]
            )  # get max entropy id
            indicator = round(proba.loc[element_id, "entropy"], 2)
            indicator = f"entropy: {indicator}"

        # get prediction of the id if it exists
        predict = {"label": None, "proba": None}

        if self.simplemodels.exists(user, scheme):
            sm = self.simplemodels.get_model(user, scheme)
            predicted_label = sm.proba.loc[element_id, "prediction"]
            predicted_proba = round(sm.proba.loc[element_id, predicted_label], 2)
            predict = {"label": predicted_label, "proba": predicted_proba}

        # get all tags already existing for the element
        history = self.schemes.db_manager.get_annotations_by_element(
            self.params.project_slug, scheme, element_id
        )

        element = {
            "element_id": element_id,
            "text": self.content.fillna("NA").loc[element_id, "text"],
            "context": dict(
                self.content.fillna("NA")
                .loc[element_id, self.params.cols_context]
                .apply(str)
            ),
            "selection": selection,
            "info": indicator,
            "predict": predict,
            "frame": frame,
            "limit": int(self.content.loc[element_id, "limit"]),
            "history": history,
        }

        return element

    def get_element(
        self,
        element_id: str,
        scheme: str | None = None,
        user: str | None = None,
        dataset: str = "train",
    ):
        """
        Get an element of the database
        TODO: better homogeneise with get_next ?
        TODO: test if element exists
        """
        if dataset == "test":
            if element_id not in self.schemes.test.index:
                return {"error": "Element does not exist !!"}
            data = {
                "element_id": element_id,
                "text": self.schemes.test.loc[element_id, "text"],
                "context": {},
                "selection": "test",
                "predict": {"label": None, "proba": None},
                "info": "",
                "frame": None,
                "limit": 1200,
                "history": [],
            }
            return data
        if dataset == "train":
            if element_id not in self.content.index:
                return {"error": "Element does not exist"}

            # get prediction if it exists
            predict = {"label": None, "proba": None}
            if (user is not None) & (scheme is not None):
                if self.simplemodels.exists(user, scheme):
                    sm = self.simplemodels.get_model(user, scheme)
                    predicted_label = sm.proba.loc[element_id, "prediction"]
                    predicted_proba = round(
                        sm.proba.loc[element_id, predicted_label], 2
                    )
                    predict = {"label": predicted_label, "proba": predicted_proba}

            # get element tags
            history = self.schemes.db_manager.get_annotations_by_element(
                self.params.project_slug, scheme, element_id
            )

            data = {
                "element_id": element_id,
                "text": self.content.loc[element_id, "text"],
                "context": dict(
                    self.content.fillna("NA")
                    .loc[element_id, self.params.cols_context]
                    .apply(str)
                ),
                "selection": "request",
                "predict": predict,
                "info": "get specific",
                "frame": None,
                "limit": int(self.content.loc[element_id, "limit"]),
                "history": history,
            }

            return data
        return {"error": "wrong set"}

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params

    def get_description(self, scheme: str | None, user: str | None):
        """
        Generate a description of a current project/scheme/user
        Return:
            JSON
        """
        if scheme is None:
            return {"error": "Scheme not defined"}

        if scheme not in self.schemes.available():
            return {"error": "Scheme does not exist"}

        # part train
        r = {"train_set_n": len(self.schemes.content)}
        r["users"] = [
            i[0]
            for i in self.db_manager.get_coding_users(scheme, self.params.project_slug)
        ]

        df = self.schemes.get_scheme_data(scheme, kind=["add", "predict"])
        r["train_annotated_n"] = len(df)
        r["train_annotated_distribution"] = json.loads(
            df["labels"].value_counts().to_json()
        )

        # part test
        if self.params.test:
            df = self.schemes.get_scheme_data(scheme, kind=["test"])
            r["test_set_n"] = len(self.schemes.test)
            r["test_annotated_n"] = len(df)
            r["test_annotated_distribution"] = json.loads(
                df["labels"].value_counts().to_json()
            )
        else:
            r["test_set_n"] = None
            r["test_annotated_n"] = None
            r["test_annotated_distribution"] = None

        if self.simplemodels.exists(user, scheme):
            sm = self.simplemodels.get_model(user, scheme)  # get model
            r["sm_10cv"] = sm.cv10

        return r

    def get_state(self):
        """
        Send state of the project
        """
        r = {
            "params": self.params,
            "users": {"active": self.get_active_users()},
            "next": {
                "methods_min": ["deterministic", "random"],
                "methods": ["deterministic", "random", "maxprob", "active"],
                "sample": ["untagged", "all", "tagged"],
            },
            "schemes": {"available": self.schemes.available(), "statistics": {}},
            "features": {
                "options": self.features.options,
                "available": list(self.features.map.keys()),
                "training": list(self.features.training.keys()),
                "infos": self.features.get_info(),
            },
            "simplemodel": {
                "options": self.simplemodels.available_models,
                "available": self.simplemodels.available(),
                "training": self.simplemodels.training(),
            },
            "bertmodels": {
                "options": self.bertmodels.base_models,
                "available": self.bertmodels.trained(),
                "training": self.bertmodels.training(),
                "test": {},
                "base_parameters": self.bertmodels.params_default,
            },
            "projections": {
                "options": self.features.possible_projections,
                # if computed : user + unique id
                "available": {
                    u: self.features.projections[u]["id"]
                    for u in self.features.projections
                    if "data" in self.features.projections[u]
                },
            },
            "generations": {"training": self.generations.generating},
        }
        return r

    def add_regex(self, name: str, value: str) -> dict:
        """
        Add regex to features
        """
        if name in self.features.map:
            return {"error": "a feature already has this name"}

        pattern = re.compile(value)
        f = self.content["text"].apply(lambda x: bool(pattern.search(x)))
        self.features.add(name, f)
        return {"success": "regex added"}

    def export_features(self, features: list, format: str = "parquet"):
        """
        Export features data in different formats
        """
        if len(features) == 0:
            return {"error": "No features selected"}

        path = self.params.dir  # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features)

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path / file_name)
        if format == "parquet":
            data.to_parquet(path / file_name)

        r = {"name": file_name, "path": path / file_name}

        return r

    def export_data(self, scheme: str, format: str = "parquet"):
        """
        Export annotation data in different formats
        """
        path = self.params.dir  # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.schemes.get_scheme_data(scheme=scheme, complete=True)

        file_name = f"data_{self.name}_{scheme}.{format}"

        # Create files
        if format == "csv":
            data.to_csv(path / file_name)
        if format == "parquet":
            data.to_parquet(path / file_name)

        r = {"name": file_name, "path": path / file_name}
        return r

    # def add_generation(
    #     self,
    #     user: str,
    #     project_name: str,
    #     endpoint: str,
    #     element_id: str,
    #     prompt: str,
    #     answer: str,
    # ):
    #     """
    #     Add a generated element in the database
    #     """
    #     self.db_manager.add_generated(
    #         user=user,
    #         project_slug=project_name,
    #         element_id=element_id,
    #         endpoint=endpoint,
    #         prompt=prompt,
    #         answer=answer,
    #     )
    #     print("Added generation", element_id)
    #     return None

    # async def generate(
    #     self, user: str, project_name: str, df: DataFrame, params: GenerateModel
    # ) -> None:
    #     """
    #     Manage generation request
    #     TODO : parallelize
    #     TODO : add other endpoint
    #     """
    #     # errors
    #     errors = []

    #     # loop on all elements
    #     for index, row in df.iterrows():
    #         # insert the content in the prompt
    #         if "#INSERTTEXT" not in params.prompt:
    #             errors.append("Problem with the prompt")
    #             continue
    #         prompt = params.prompt.replace("#INSERTTEXT", row["text"])

    #         # make request to the client
    #         if params.api == "ollama":
    #             response = await functions.request_ollama(params.endpoint, prompt)
    #         else:
    #             errors.append("Model does not exist")
    #             continue

    #         if "error" in response:
    #             errors.append("Error in the request " + response["error"])

    #         if "success" in response:
    #             self.add_generation(
    #                 user,
    #                 project_name,
    #                 params.endpoint,
    #                 row["index"],
    #                 prompt,
    #                 response["success"],
    #             )

    #         print("element generated ", index)

    #     print(errors)
    #     return None

    # async def compute_zeroshot(self, df, params):
    #     """
    #     Zero-shot beta version
    #     # TODO : chunk & control the context size
    #     """
    #     r_error = ["error"] * len(df)

    #     # create the chunks
    #     # FOR THE MOMENT, ONLY 10 elements for DEMO
    #     if len(df) > 10:
    #         df = df[0:10]

    #     # create prompt
    #     list_texts = "\nTexts to annotate:\n"
    #     for i, t in enumerate(list(df["text"])):
    #         list_texts += f"{i}. {t}\n"
    #     prompt = (
    #         params.prompt
    #         + list_texts
    #         + '\nResponse format:\n{"annotations": [{"text": "Text 1", "label": "Label1"}, {"text": "Text 2", "label": "Label2"}, ...]}'
    #     )

    #     # make request to client
    #     # client = openai.OpenAI(api_key=params.token)
    #     try:
    #         self.zeroshot = "computing"
    #         client = openai.AsyncOpenAI(api_key=params.token)
    #         print("Make openai call")
    #         chat_completion = await client.chat.completions.create(
    #             messages=[
    #                 {
    #                     "role": "system",
    #                     "content": """Your are a careful assistant who annotates texts for a research project.
    #                 You follow precisely the guidelines, which can be in different languages.
    #                 """,
    #                 },
    #                 {
    #                     "role": "user",
    #                     "content": prompt,
    #                 },
    #             ],
    #             model="gpt-3.5-turbo",
    #             response_format={"type": "json_object"},
    #         )
    #         print("OpenAI call done")
    #     except:
    #         self.zeroshot = None
    #         return {"error": "API connexion failed. Check the token."}
    #     # extracting results
    #     try:
    #         r = json.loads(chat_completion.choices[0].message.content)["annotations"]
    #         r = [i["label"] for i in r]
    #     except:
    #         return {"error": "Format problem"}
    #     if len(r) == len(df):
    #         df["zero_shot"] = r
    #         self.zeroshot = df[["text", "zero_shot"]].reset_index().to_json()
    #         return {"success": "data computed"}
    #     else:
    #         return {"error": "Problem with the number of element"}

    def get_active_users(self, period: int = 300):
        """
        Get current active users on the time period
        """
        users = self.db_manager.get_distinct_users(self.name, period)
        return users
