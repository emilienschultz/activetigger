import json
import logging
import os
import secrets
import shutil
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import yaml
from fastapi.encoders import jsonable_encoder
from jose import jwt
from pandas import DataFrame
from pydantic import ValidationError
from slugify import slugify

from activetigger.datamodels import (
    ProjectDataModel,
    ProjectModel,
    ProjectSummaryModel,
    SimpleModelModel,
    TestSetDataModel,
)
from activetigger.db import DatabaseManager
from activetigger.features import Features
from activetigger.functions import clean_regex
from activetigger.generations import Generations
from activetigger.models import BertModels, SimpleModels
from activetigger.projections import Projections
from activetigger.queue import Queue
from activetigger.schemes import Schemes
from activetigger.users import Users

logger = logging.getLogger("server")

# Define server parameters
db_name = "activetigger.db"
data_all = "data_all.parquet"
features_file = "features.parquet"
annotations_file = "annotations.parquet"
train_file = "train.parquet"
test_file = "test.parquet"
default_user = "root"
ALGORITHM = "HS256"
MAX_LOADED_PROJECTS = 20


class Server:
    """
    Server to manage backend
    """

    db_name: str
    features_file: str
    annotations_file: str
    data_all: str
    train_file: str
    test_file: str
    default_user: str
    ALGORITHM: str
    n_workers: int = 4
    starting_time: float = None
    SECRET_KEY: str
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
        self.db_name = db_name
        self.data_all = data_all
        self.features_file = features_file
        self.annotations_file = annotations_file
        self.train_file = train_file
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

        # logging
        logging.basicConfig(
            filename=self.path / "log_server.log",
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

    def get_logs(
        self, username: str, project_slug: str, limit: int, partial: bool = True
    ) -> pd.DataFrame:
        """
        Get logs for a user/project
        """
        logs = self.db_manager.get_logs("all", project_slug, limit)
        df = pd.DataFrame(
            logs, columns=["id", "time", "user", "project", "action", "NA"]
        )
        if partial:
            return df[~df["action"].str.contains("INFO ")]
        return df

    def get_auth_projects(self, username: str) -> dict[dict]:
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
        Get project params from database
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

        self.projects[project_slug] = Project(
            project_slug, self.queue, self.db_manager, path_models=self.path_models
        )
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

    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        existing_projects = self.db_manager.existing_projects()
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
            return {"error": "Project name already exist"}

        # test if the name of the column is specified
        if params.col_id is None or params.col_id == "":
            return {"error": "Probleme with the id column: empty name"}
        if params.col_text is None or params.col_text == "":
            return {"error": "Probleme with the text column: empty name"}

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
        content = pd.read_csv(params.dir / "data_raw.csv", dtype=str)
        # quick fix to avoid problem with pandas index saved in parquet
        content = content.drop(
            columns=[i for i in content.columns if "__index_level" in i]
        )
        # remove empty lines
        content = content.dropna(how="all")
        all_columns = list(content.columns)
        n_total = len(content)

        # test if the size of the sample requested is possible
        if len(content) < params.n_test + params.n_train:
            shutil.rmtree(params.dir)
            return {
                "error": f"Not enough data for creating the train/test dataset. Current : {len(content)} ; Selected : {params.n_test + params.n_train}"
            }

        # check if index after slugify is unique otherwise FORCE the index from 0 to N
        if not (
            (content[params.col_id].astype(str).apply(slugify)).nunique()
            == len(content)
        ):
            print("There are duplicate in the column selected for index")
            content["id"] = range(0, len(content))
            params.col_id = "id"

        # rename the index col, transform it in str, and set it as index
        if "id" in content.columns:
            content["id_raw"] = content["id"]  # copy the column id to not erase it
            # TODO : take into account the fact that id column can exist
        content["id"] = content[params.col_id].astype(str).apply(slugify)
        content.set_index("id", inplace=True)

        # create the text column, merging the different columns
        if isinstance(params.col_text, list):
            content["text"] = content[params.col_text].apply(
                lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
            )
        else:
            content["text"] = content[params.col_text]

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
        content.to_parquet(params.dir / self.data_all, index=True)

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

        trainset.to_parquet(params.dir / self.train_file, index=True)
        trainset[list(set(["text"] + params.cols_context))].to_parquet(
            params.dir / self.annotations_file, index=True
        )
        trainset[[]].to_parquet(params.dir / self.features_file, index=True)

        # if the case, add existing annotations in the database
        if params.col_label is None:
            self.db_manager.add_scheme(project_slug, "default", json.dumps([]), "file")
        else:
            # check there is a limited number of labels

            df = content["label"].dropna()
            params.default_scheme = list(df.unique())

            if len(params.default_scheme) < 30:
                print("Add scheme/labels from file in train/test")

                # add the scheme in the database
                self.db_manager.add_scheme(
                    project_slug, "default", json.dumps(params.default_scheme), "file"
                )

                # add the labels from the trainset in the database
                for element_id, label in trainset["label"].dropna().items():
                    self.db_manager.add_annotation(
                        dataset="train",
                        user=username,
                        project_slug=project_slug,
                        element_id=element_id,
                        scheme="default",
                        annotation=label,
                    )
                    print("add annotations train", element_id)
                # add the labels from the trainset in the database if exists & not clear
                if isinstance(testset, pd.DataFrame) and not params.clear_test:
                    for element_id, label in testset["label"].dropna().items():
                        self.db_manager.add_annotation(
                            dataset="test",
                            user=username,
                            project_slug=project_slug,
                            element_id=element_id,
                            scheme="default",
                            annotation=label,
                        )
                        print("add annotations test", element_id)
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
        os.remove(params.dir / "data_raw.csv")

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


class Project(Server):
    """
    Project object
    """

    starting_time: float
    name: str
    queue: Queue
    computing: list
    path_models: Path
    db_manager: DatabaseManager
    params: ProjectModel
    content: DataFrame
    schemes: Schemes
    features: Features
    bertmodels: BertModels
    simplemodels: SimpleModels
    generations: Generations
    projections: Projections

    def __init__(
        self,
        project_slug: str,
        queue: Queue,
        db_manager: DatabaseManager,
        path_models: Path,
    ) -> None:
        """
        Load existing project
        """
        self.starting_time = time.time()
        self.name = project_slug
        self.queue = queue
        self.computing = []  # currently computing elements
        self.db_manager = db_manager
        self.params = self.load_params(project_slug)
        self.path_models = path_models

        # check if directory exists
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # loading data
        self.content = pd.read_parquet(self.params.dir / train_file)

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.params.dir / annotations_file,
            self.params.dir / test_file,
            self.db_manager,
        )
        self.features = Features(
            project_slug,
            self.params.dir / features_file,
            self.params.dir / data_all,
            self.params.dir / self.path_models,
            self.queue,
            self.computing,
            self.db_manager,
            self.params.language,
        )
        self.bertmodels = BertModels(
            self.params.dir, self.queue, self.computing, self.db_manager
        )
        self.simplemodels = SimpleModels(self.params.dir, self.queue, self.computing)
        self.generations = Generations(self.db_manager, self.computing)
        self.projections = Projections(self.computing)

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

    def update_simplemodel(self, simplemodel: SimpleModelModel, username: str) -> dict:
        """
        Update simplemodel on the base of an already existing
        simplemodel object

        n_min: minimal number of elements annotated
        """
        if simplemodel.features is None or len(simplemodel.features) == 0:
            return {"error": "No feature selected"}
        if simplemodel.model not in list(self.simplemodels.available_models.keys()):
            return {"error": "Model doesn't exist"}
        if simplemodel.scheme not in self.schemes.available():
            return {"error": "Scheme doesn't exist"}
        if len(self.schemes.available()[simplemodel.scheme]) < 2:
            return {"error": "2 different labels needed"}

        # only dfm feature for multi_naivebayes (FORCE IT if available else error)
        if simplemodel.model == "multi_naivebayes":
            if "dfm" not in self.features.map:
                return {"error": "dfm feature not available for multi_naivebayes"}
            simplemodel.features = ["dfm"]
            simplemodel.standardize = False

        # test if the parameters have the correct format
        try:
            validation = self.simplemodels.validation[simplemodel.model]
            params = validation(**simplemodel.params).dict()
        except ValidationError as e:
            return {"error": e.json()}

        # get data
        df_features = self.features.get(simplemodel.features)
        df_scheme = self.schemes.get_scheme_data(scheme=simplemodel.scheme)

        # test for a minimum of annotated elements
        counts = df_scheme["labels"].value_counts()
        valid_categories = counts[counts >= 3]
        if len(valid_categories) < 2:
            return {
                "error": "there are less than 2 categories with 3 annotated elements"
            }

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)

        logger_simplemodel = logging.getLogger("simplemodel")
        logger_simplemodel.info("Building the simplemodel request")
        self.simplemodels.compute_simplemodel(
            user=username,
            scheme=simplemodel.scheme,
            features=simplemodel.features,
            name=simplemodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=params,
            standardize=simplemodel.standardize,
        )

        return {"success": "Simplemodel updated"}

    def get_next(
        self,
        scheme: str,
        selection: str = "deterministic",
        sample: str = "untagged",
        user: str = "user",
        label: None | str = None,
        history: list = [],
        frame: None | list = None,
        filter: str | None = None,
    ) -> dict:
        """
        Get next item for a specific scheme with a specific selection method
        - deterministic
        - random
        - active
        - maxprob
        - test

        history : previous selected elements
        frame is the use of projection coordinates to limit the selection
        filter is a regex to use on the corpus
        """

        if scheme not in self.schemes.available():
            return {"error": "Scheme doesn't exist"}

        # size of the subsample

        # specific case of test, random element
        if selection == "test":
            df = self.schemes.get_scheme_data(scheme, complete=True, kind=["test"])
            f = df["labels"].isnull()
            if len(df[f]) == 0:
                return {"error": "No element to annotate"}
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
                "n_sample": f.sum(),
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
            # sanitize
            filter_san = clean_regex(filter)
            if "CONTEXT=" in filter_san:  # case to search in the context
                f_regex = (
                    df[self.params.cols_context]
                    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
                    .str.contains(
                        filter_san.replace("CONTEXT=", ""),
                        regex=True,
                        case=True,
                        na=False,
                    )
                )
            else:
                f_regex = df["text"].str.contains(
                    filter_san, regex=True, case=True, na=False
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

        n_sample = len(ss)

        if selection == "deterministic":  # next row
            element_id = ss.index[0]

        if selection == "random":  # random row
            element_id = ss.sample(frac=1).index[0]

        # higher prob, only possible if the model has been trained
        if selection == "maxprob":
            if not self.simplemodels.exists(user, scheme):
                return {"error": "Simplemodel doesn't exist"}
            if label is None:  # default label to first
                return {"error": "Select a tag"}
            sm = self.simplemodels.get_model(user, scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            ss = (
                proba[f][label]
                .drop(history, errors="ignore")
                .sort_values(ascending=False)
            )  # get max proba id
            element_id = ss.index[0]
            n_sample = len(ss)
            indicator = f"probability: {round(proba.loc[element_id,label],2)}"

        # higher entropy, only possible if the model has been trained
        if selection == "active":
            if not self.simplemodels.exists(user, scheme):
                return {"error": "Simplemodel doesn't exist"}
            sm = self.simplemodels.get_model(user, scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            ss = (
                proba[f]["entropy"]
                .drop(history, errors="ignore")
                .sort_values(ascending=False)
            )  # get max entropy id
            element_id = ss.index[0]
            n_sample = len(ss)
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
        previous = self.schemes.db_manager.get_annotations_by_element(
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
            "history": previous,
            "n_sample": n_sample,
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
        Separate train/test dataset
        TODO: better homogeneise with get_next ?
        """
        if dataset == "test":
            if element_id not in self.schemes.test.index:
                return {"error": "Element does not exist."}
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
                return {"error": "Element does not exist."}

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

    def get_statistics(self, scheme: str | None, user: str | None):
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

        df = self.schemes.get_scheme_data(scheme, kind=["train", "predict"])
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
        # start_time = time.time()
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
                "training": self.features.current_computing(),
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
                "options": self.projections.options,
                "available": {
                    i: self.projections.available[i]["id"]
                    for i in self.projections.available
                },
                "training": self.projections.training(),  # list(self.projections.training().keys()),
            },
            "generations": {"training": self.generations.current_users_generating()},
        }
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time:.5f} seconds")
        return r

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
        if format == "xlsx":
            data.to_excel(path / file_name)

        r = {"name": file_name, "path": path / file_name}

        return r

    def export_data(self, scheme: str, dataset: str = "train", format: str = "parquet"):
        """
        Export annotation data in different formats
        """
        path = self.params.dir  # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        # test or train
        if dataset == "test":
            if not self.params.test:
                return {"error": "No test data"}
            data = self.schemes.get_scheme_data(
                scheme=scheme, complete=True, kind="test"
            )
            file_name = f"data_test_{self.name}_{scheme}.{format}"
        else:
            data = self.schemes.get_scheme_data(scheme=scheme, complete=True)
            file_name = f"data_train_{self.name}_{scheme}.{format}"

        # Create files
        if format == "csv":
            data.reset_index().map(str).to_csv(path / file_name)
        if format == "parquet":
            data.reset_index().map(str).to_parquet(path / file_name)
        if format == "xlsx":
            data.reset_index().map(str).to_excel(path / file_name)

        r = {"name": file_name, "path": path / file_name}
        return r

    def get_active_users(self, period: int = 300):
        """
        Get current active users on the time period
        """
        users = self.db_manager.get_distinct_users(self.name, period)
        return users

    def get_process(self, kind: str, user: str):
        """
        Get current processes
        """
        return [e for e in self.computing if e["user"] == user and e["kind"] == kind]

    def update_processes(self) -> dict:
        """
        Update completed processes and do specific operations regarding their kind
        - get the result from the queue
        - add the result if needed
        """
        predictions = {}

        for e in self.computing.copy():
            # clean flag
            clean = False

            # case of not in queue
            if e["unique_id"] not in self.queue.current:
                print("Problem : id in computing not in queue")
                self.computing.remove(e)
                continue

            is_done = self.queue.current[e["unique_id"]]["future"].done()

            # case for bertmodels
            if (e["kind"] == "bert") and is_done:
                clean = True
                try:
                    if e["model"].status == "training":
                        print("Model trained")
                    if e["model"].status == "testing":
                        print("Model tested")
                    if e["model"].status == "predicting train":
                        print("Prediction train finished")
                        df = self.queue.current[e["unique_id"]]["future"].result()
                        predictions["predict_" + e["model"].name] = df["prediction"]
                except Exception as ex:
                    print("Error in model training/predicting", ex)

            # case for simplemodels
            if (e["kind"] == "simplemodel") and is_done:
                clean = True
                try:
                    results = self.queue.current[e["unique_id"]]["future"].result()
                    self.simplemodels.add(e, results)
                    print("Simplemodel trained")
                except Exception as ex:
                    print("Simplemodel failed", ex)

            # case for features
            if (e["kind"] == "feature") and is_done:
                clean = True
                try:
                    r = self.queue.current[e["unique_id"]]["future"].result()
                    self.features.add(
                        e["name"], e["type"], e["user"], e["parameters"], r["success"]
                    )
                    print("Feature added", e["name"])
                except Exception as ex:
                    print("Error in feature processing", ex)

            # case for projections
            if (e["kind"] == "projection") and is_done:
                clean = True
                try:
                    df = self.queue.current[e["unique_id"]]["future"].result()
                    self.projections.add(e, df)
                    print("projection added")
                except Exception as ex:
                    print("Error in feature projections queue", ex)

            # case for generations
            if (e["kind"] == "generation") and is_done:
                clean = True
                try:
                    r = self.queue.current[e["unique_id"]]["future"].result()
                    for row in r["success"]:
                        self.generations.add(
                            user=row["user"],
                            project_slug=row["project_slug"],
                            element_id=row["element_id"],
                            endpoint=row["endpoint"],
                            prompt=row["prompt"],
                            answer=row["answer"],
                        )
                except Exception as ex:
                    print("Error in generation queue", ex)

            # delete from computing & queue
            if clean:
                self.computing.remove(e)
                self.queue.delete(e["unique_id"])

        return predictions
