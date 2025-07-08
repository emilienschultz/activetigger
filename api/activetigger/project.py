import io
import json
import logging
import os
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pandas as pd
import pytz
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pandas import DataFrame

from activetigger.config import config
from activetigger.datamodels import (
    BertModelModel,
    ElementOutModel,
    ExportGenerationsParams,
    FeatureComputing,
    GenerationComputing,
    GenerationModel,
    GenerationRequest,
    GenerationResult,
    LMComputing,
    NextInModel,
    NextProjectStateModel,
    ProjectionComputing,
    ProjectionOutModel,
    ProjectModel,
    ProjectStateModel,
    ProjectUpdateModel,
    SimpleModelComputing,
    SimpleModelModel,
    StaticFileModel,
    TestSetDataModel,
)
from activetigger.db.manager import DatabaseManager
from activetigger.features import Features
from activetigger.functions import clean_regex, get_dir_size, slugify
from activetigger.generation.generations import Generations
from activetigger.languagemodels import LanguageModels
from activetigger.projections import Projections
from activetigger.queue import Queue
from activetigger.schemes import Schemes
from activetigger.simplemodels import SimpleModels
from activetigger.tasks.generate_call import GenerateCall

MODELS = "bert_models.csv"
TIMEZONE = pytz.timezone("Europe/Paris")


class Project:
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
    languagemodels: LanguageModels
    simplemodels: SimpleModels
    generations: Generations
    projections: Projections
    errors: list[list]

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
        self.queue = queue
        self.computing = []
        self.db_manager = db_manager
        self.path_models = path_models

        # load the project
        self.name = project_slug
        self.load_project(project_slug)

    def __del__(self):
        pass

    def delete(self):
        """
        Delete completely a project
        """
        # remove from database
        try:
            self.db_manager.projects_service.delete_project(self.params.project_slug)
        except Exception as e:
            raise ValueError("Problem with the database " + str(e))

        # remove folder of the project
        try:
            shutil.rmtree(self.params.dir)
        except Exception as e:
            raise ValueError("No directory to delete " + str(e))

        # remove static files
        if Path(f"{config.data_path}/projects/static/{self.name}").exists():
            shutil.rmtree(f"{config.data_path}/projects/static/{self.name}")

    def load_project(self, project_slug: str):
        """
        Load existing project
        """

        try:
            self.params = self.load_params(project_slug)
        except Exception as e:
            raise ValueError("This project can be loaded", str(e))

        # check if directory exists
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # loading data
        self.content = pd.read_parquet(self.params.dir.joinpath("train.parquet"))

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.params.dir.joinpath("train.parquet"),
            self.params.dir.joinpath("test.parquet"),
            self.db_manager,
        )
        self.features = Features(
            project_slug,
            self.params.dir.joinpath("features.parquet"),
            self.params.dir.joinpath("data_all.parquet"),
            self.path_models,
            self.queue,
            cast(list[FeatureComputing], self.computing),
            self.db_manager,
            self.params.language,
        )
        self.languagemodels = LanguageModels(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.db_manager,
            MODELS,
        )
        self.simplemodels = SimpleModels(self.params.dir, self.queue, self.computing)
        # TODO: Computings should be filtered here based on their type, each type in a different list given to the appropriate class.
        # It would render the cast here and the for-loop in the class unecessary
        self.generations = Generations(
            self.db_manager, cast(list[GenerationComputing], self.computing)
        )
        self.projections = Projections(self.params.dir, self.computing, self.queue)
        self.errors = []  # Move to specific class / db in the future

    def load_params(self, project_slug: str) -> ProjectModel:
        """
        Load params from database
        """
        existing_project = self.db_manager.projects_service.get_project(project_slug)
        if existing_project:
            return ProjectModel(**existing_project["parameters"])
        else:
            raise NameError(f"{project_slug} does not exist.")

    def drop_testset(self) -> None:
        """
        Clean all the test data of the project
        - remove the file
        - remove all the annotations in the database
        - set the flag to False
        """
        if not self.params.dir:
            raise Exception("No directory for project")
        path_testset = self.params.dir.joinpath("test.parquet")
        if not path_testset.exists():
            raise Exception("No test data available")
        os.remove(path_testset)
        self.db_manager.projects_service.delete_annotations_testset(self.params.project_slug)
        self.schemes.test = None
        self.params.test = False
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

    def add_testset(self, testset: TestSetDataModel, username: str, project_slug: str) -> None:
        """
        Add a test dataset

        The test dataset should :
        - not contains NA
        - have a unique id different from the complete dataset

        The id will be modified to indicate imported

        """
        if self.schemes.test is not None:
            raise Exception("There is already a test dataset")

        if self.params.dir is None:
            raise Exception("Cannot add test data without a valid dir")

        if testset.col_label == "":
            testset.col_label = None

        csv_buffer = io.StringIO(testset.csv)
        df = pd.read_csv(
            csv_buffer,
            dtype={testset.col_id: str, testset.col_text: str},
            nrows=testset.n_test,
        )

        if len(df) > 10000:
            raise Exception("You testset is too large")

        # change names
        if not testset.col_label:
            df = df.rename(columns={testset.col_id: "id", testset.col_text: "text"})
        else:
            df = df.rename(
                columns={
                    testset.col_id: "id",
                    testset.col_text: "text",
                    testset.col_label: "label",
                }
            )

        # deal with non-unique id
        # TODO : compare with the general dataset ???
        if not ((df["id"].astype(str).apply(slugify)).nunique() == len(df)):
            df["id"] = [str(i) for i in range(len(df))]
            print("ID not unique, changed to default id")

        # identify the dataset as imported and set the id
        df["id"] = df["id"].apply(lambda x: f"imported-{x}")
        df = df.set_index("id")

        # import labels if specified + scheme // check if the labels are in the scheme
        if testset.col_label and testset.scheme:
            # Check the label columns if they match the scheme or raise error
            scheme = self.schemes.available()[testset.scheme]["labels"]
            for label in df["label"].dropna().unique():
                if label not in scheme:
                    raise Exception(f"Label {label} not in the scheme {testset.scheme}")

            elements = [
                {"element_id": element_id, "annotation": label, "comment": ""}
                for element_id, label in df["label"].dropna().items()
            ]
            self.db_manager.projects_service.add_annotations(
                dataset="test",
                user_name=username,
                project_slug=project_slug,
                scheme=testset.scheme,
                elements=elements,
            )
            print("Testset labels imported")

        # write the dataset
        df[["text"]].to_parquet(self.params.dir.joinpath("test.parquet"))
        # load the data
        self.schemes.test = df[["text"]]
        # update parameters
        self.params.test = True
        # update the database
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

    def update_simplemodel(self, simplemodel: SimpleModelModel, username: str) -> dict:
        """
        Update simplemodel on the base of an already existing
        simplemodel object

        n_min: minimal number of elements annotated
        """
        availabe_schemes = self.schemes.available()
        simplemodel.features = [i for i in simplemodel.features if i is not None]
        if simplemodel.features is None or len(simplemodel.features) == 0:
            raise Exception("No features selected")
        if simplemodel.model not in list(self.simplemodels.available_models.keys()):
            raise Exception("Model not available")
        if simplemodel.scheme not in availabe_schemes:
            raise Exception("Scheme not available")
        if len(availabe_schemes[simplemodel.scheme].labels) < 2:
            raise Exception("Scheme not available")

        # only dfm feature for multi_naivebayes (FORCE IT if available else error)
        if simplemodel.model == "multi_naivebayes":
            if "dfm" not in self.features.map:
                raise Exception("No dfm feature available")
            simplemodel.features = ["dfm"]
            simplemodel.standardize = False

        if simplemodel.params is None:
            params = None
        else:
            params = dict(simplemodel.params)
        # add information on the target of the model
        if simplemodel.dichotomize is not None and params is not None:
            params["dichotomize"] = simplemodel.dichotomize

        # get data
        df_features = self.features.get(simplemodel.features)
        df_scheme = self.schemes.get_scheme_data(scheme=simplemodel.scheme)

        # management for multilabels / dichotomize
        if simplemodel.dichotomize is not None:
            df_scheme["labels"] = df_scheme["labels"].apply(
                lambda x: self.schemes.dichotomize(x, simplemodel.dichotomize)
            )

        # test for a minimum of annotated elements
        counts = df_scheme["labels"].value_counts()
        valid_categories = counts[counts >= 3]
        if len(valid_categories) < 2:
            raise Exception("Not enough annotated elements")

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)

        logger_simplemodel = logging.getLogger("simplemodel")
        logger_simplemodel.info("Building the simplemodel request")
        self.simplemodels.compute_simplemodel(
            project_slug=self.params.project_slug,
            user=username,
            scheme=simplemodel.scheme,
            features=simplemodel.features,
            name=simplemodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=params,
            standardize=simplemodel.standardize or False,
            cv10=simplemodel.cv10 or False,
        )

        return {"success": "Simplemodel updated"}

    def get_next(
        self,
        next: NextInModel,
        username: str = "user",
    ) -> ElementOutModel:
        """
        Get next item for a specific scheme with a specific selection method
        - fixed
        - random
        - active
        - maxprob
        - test

        history : previous selected elements
        frame is the use of projection coordinates to limit the selection
        filter is a regex to use on the corpus
        """

        if next.scheme not in self.schemes.available():
            raise ValueError("Scheme doesn't exist")

        # select the current state of annotation
        if next.dataset == "test":
            df = self.schemes.get_scheme_data(next.scheme, complete=True, kind=["test"])
        else:
            df = self.schemes.get_scheme_data(next.scheme, complete=True)

        # build first filter from the sample
        if next.sample == "untagged":
            f = df["labels"].isna()
        elif next.sample == "tagged":
            if next.label is not None and next.label in df["labels"].unique():
                f = df["labels"] == next.label
            else:
                f = df["labels"].notna()
        else:
            f = df["labels"].apply(lambda x: True)

        # add a regex condition to the selection
        if next.filter:
            # sanitize
            df["ID"] = df.index  # duplicate the id column
            filter_san = clean_regex(next.filter)
            if "CONTEXT=" in filter_san:  # case to search in the context
                print("CONTEXT", filter_san.replace("CONTEXT=", ""))
                f_regex: pd.Series = (
                    df[self.params.cols_context + ["ID"]]
                    .apply(lambda row: " ".join(row.values.astype(str)), axis=1)
                    .str.contains(
                        filter_san.replace("CONTEXT=", ""),
                        regex=True,
                        case=True,
                        na=False,
                    )
                )
            elif "QUERY=" in filter_san:  # case to use a query
                f_regex = df[self.params.cols_context].eval(filter_san.replace("QUERY=", ""))
            else:
                f_regex = df["text"].str.contains(filter_san, regex=True, case=True, na=False)
            f = f & f_regex

        # manage frame selection (if projection, only in the box)
        if next.frame and len(next.frame) == 4:
            if username in self.projections.available:
                if self.projections.available[username].data:
                    projection = self.projections.available[username].data
                    f_frame = (
                        (projection[0] > next.frame[0])
                        & (projection[0] < next.frame[1])
                        & (projection[1] > next.frame[2])
                        & (projection[1] < next.frame[3])
                    )
                    f = f & f_frame
                else:
                    raise ValueError("No vizualisation data available")
            else:
                raise ValueError("No vizualisation available")

        # test if there is at least one element available
        if sum(f) == 0:
            raise ValueError("No element available with this selection mode.")

        # Take into account the session history
        ss = df[f].drop(next.history, errors="ignore")
        if len(ss) == 0:
            raise ValueError("No element available with this selection mode.")
        indicator = None
        n_sample = f.sum()  # use len(ss) for adding history

        # select type of selection
        if next.selection == "fixed":  # next row
            element_id = ss.index[0]

        if next.selection == "random":  # random row
            element_id = ss.sample(frac=1).index[0]

        # higher prob for the label_maxprob, only possible if the model has been trained
        if next.selection == "maxprob":
            if not self.simplemodels.exists(username, next.scheme):
                raise Exception("Simplemodel doesn't exist")
            if next.label_maxprob is None:  # default label to first
                raise Exception("Label maxprob is required")
            sm = self.simplemodels.get_model(username, next.scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            ss = (
                proba[f][next.label_maxprob]
                .drop(next.history, errors="ignore")
                .sort_values(ascending=False)
            )  # get max proba id
            element_id = ss.index[0]
            n_sample = f.sum()
            indicator = f"probability: {round(proba.loc[element_id, next.label_maxprob], 2)}"

        # higher entropy, only possible if the model has been trained
        if next.selection == "active":
            if not self.simplemodels.exists(username, next.scheme):
                raise ValueError("Simplemodel doesn't exist")
            sm = self.simplemodels.get_model(username, next.scheme)  # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            ss = (
                proba[f]["entropy"].drop(next.history, errors="ignore").sort_values(ascending=False)
            )  # get max entropy id
            element_id = ss.index[0]
            n_sample = f.sum()
            indicator = round(proba.loc[element_id, "entropy"], 2)
            indicator = f"entropy: {indicator}"

        # get prediction of the id if it exists
        predict = {"label": None, "proba": None}

        if self.simplemodels.exists(username, next.scheme) and next.dataset == "train":
            sm = self.simplemodels.get_model(username, next.scheme)
            predicted_label = sm.proba.loc[element_id, "prediction"]
            predicted_proba = round(sm.proba.loc[element_id, predicted_label], 2)
            predict = {"label": predicted_label, "proba": predicted_proba}

        # get all tags already existing for the element
        previous = self.schemes.projects_service.get_annotations_by_element(
            self.params.project_slug, next.scheme, element_id
        )
        if next.dataset == "test":
            limit = 1200
            context = {}
        else:
            limit = int(self.content.loc[element_id, "limit"])
            # get context
            context = dict(
                self.content.fillna("NA").loc[element_id, self.params.cols_context].apply(str)
            )

        return ElementOutModel(
            element_id=element_id,
            text=df.fillna("NA").loc[element_id, "text"],
            context=context,
            selection=next.selection,
            info=indicator,
            predict=predict,
            frame=next.frame,
            limit=limit,
            history=previous,
            n_sample=n_sample,
        )

    def get_element(
        self,
        element_id: str,
        scheme: str | None = None,
        user: str | None = None,
        dataset: str = "train",
    ) -> ElementOutModel:
        """
        Get an element of the database
        Separate train/test dataset
        TODO: better homogeneise with get_next ?
        """
        if dataset == "test" and self.schemes.test is not None:
            if element_id not in self.schemes.test.index:
                raise Exception("Element does not exist.")
            return ElementOutModel(
                element_id=element_id,
                text=self.schemes.test.loc[element_id, "text"],
                context={},
                selection="test",
                info="",
                predict={"label": None, "proba": None},
                frame=None,
                limit=1200,
                history=[],
            )

        if dataset == "train":
            if element_id not in self.content.index:
                raise Exception("Element does not exist.")

            # get prediction if it exists
            predict = {"label": None, "proba": None}
            if (user is not None) and (scheme is not None):
                if self.simplemodels.exists(user, scheme):
                    sm = self.simplemodels.get_model(user, scheme)
                    predicted_label = sm.proba.loc[element_id, "prediction"]
                    predicted_proba = round(sm.proba.loc[element_id, predicted_label], 2)
                    predict = {"label": predicted_label, "proba": predicted_proba}

            # get element tags
            history = self.schemes.projects_service.get_annotations_by_element(
                self.params.project_slug, scheme, element_id
            )

            return ElementOutModel(
                element_id=element_id,
                text=self.content.loc[element_id, "text"],
                context=dict(
                    self.content.fillna("NA").loc[element_id, self.params.cols_context].apply(str)
                ),
                selection="request",
                predict=predict,
                info="get specific",
                frame=None,
                limit=int(self.content.loc[element_id, "limit"]),
                history=history,
            )

        raise Exception("Dataset does not exist.")

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params

    def get_statistics(self, scheme: str | None, user: str | None) -> dict:
        """
        Generate a description of a current project/scheme/user
        Return:
            JSON
        """
        if scheme is None:
            raise Exception("Scheme is required")

        schemes = self.schemes.available()
        if scheme not in schemes:
            raise Exception("Scheme not available")
        kind = schemes[scheme].kind

        # part train
        r = {"train_set_n": len(self.schemes.content)}
        r["users"] = self.db_manager.users_service.get_coding_users(
            scheme, self.params.project_slug
        )

        df = self.schemes.get_scheme_data(scheme, kind=["train", "predict"])

        # different treatment if the scheme is multilabel or multiclass
        r["train_annotated_n"] = len(df.dropna(subset=["labels"]))
        if kind == "multiclass":
            r["train_annotated_distribution"] = json.loads(df["labels"].value_counts().to_json())
        else:
            r["train_annotated_distribution"] = json.loads(
                df["labels"].str.split("|").explode().value_counts().to_json()
            )

        # part test
        if self.params.test:
            df = self.schemes.get_scheme_data(scheme, kind=["test"])
            r["test_set_n"] = len(self.schemes.test) if self.schemes.test is not None else 0
            r["test_annotated_n"] = len(df.dropna(subset=["labels"]))
            if kind == "multiclass":
                r["test_annotated_distribution"] = json.loads(df["labels"].value_counts().to_json())
            else:
                r["test_annotated_distribution"] = json.loads(
                    df["labels"].str.split("|").explode().value_counts().to_json()
                )
        else:
            r["test_set_n"] = None
            r["test_annotated_n"] = None
            r["test_annotated_distribution"] = None

        if self.simplemodels.exists(user, scheme):
            sm = self.simplemodels.get_model(user, scheme)  # get model
            r["sm_10cv"] = sm.statistics_cv10

        return r

    def get_projection(self, username: str, scheme: str) -> ProjectionOutModel | None:
        """
        Get projection if computed
        """
        projection = self.projections.get(username)
        if projection is None:
            return None
        # get annotations
        df = self.schemes.get_scheme_data(scheme, complete=True)
        data = projection.data
        data["labels"] = df["labels"].fillna("NA")

        # get & add predictions if available
        if username in self.simplemodels.existing:
            if scheme in self.simplemodels.existing[username]:
                data["prediction"] = self.simplemodels.existing[username][scheme].proba[
                    "prediction"
                ]

        return ProjectionOutModel(
            index=list(data.index),
            x=list(data[0]),
            y=list(data[1]),
            status=projection.id,
            parameters=projection.parameters,
            labels=list(data["labels"]),
            predictions=list(data["prediction"]) if "prediction" in data else None,
        )

    def state(self) -> ProjectStateModel:
        """
        Send state of the project
        """

        return ProjectStateModel(
            params=self.params,
            next=NextProjectStateModel(
                methods_min=["fixed", "random"],
                methods=["fixed", "random", "maxprob", "active"],
                sample=["untagged", "all", "tagged"],
            ),
            schemes=self.schemes.state(),
            features=self.features.state(),
            simplemodel=self.simplemodels.state(),
            languagemodels=self.languagemodels.state(),
            projections=self.projections.state(),
            generations=self.generations.state(),
            errors=self.errors,
            memory=get_dir_size(str(self.params.dir)),
            last_activity=self.db_manager.logs_service.get_last_activity_project(
                self.params.project_slug
            ),
        )

    def export_features(self, features: list, format: str = "parquet") -> FileResponse:
        """
        Export features data in different formats
        """
        if len(features) == 0:
            raise ValueError("No feature selected")

        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features)

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path=path.joinpath(file_name), name=file_name)

    def export_data(
        self, scheme: str, dataset: str = "train", format: str = "parquet", dropna: bool = True
    ) -> FileResponse:
        """
        Export annotation data in different formats
        """
        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        # test or train
        if dataset == "test":
            if not self.params.test:
                raise Exception("No test data available")
            data = self.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["test"])
            file_name = f"data_test_{self.name}_{scheme}.{format}"
        else:
            data = self.schemes.get_scheme_data(scheme=scheme, complete=True)
            file_name = f"data_train_{self.name}_{scheme}.{format}"

        # transformation
        if dropna:
            data = data.dropna(subset=["labels"])
        data = (
            data.rename(columns={"labels": scheme, "id": "at_id"})
            .drop(columns=["limit"], errors="ignore")
            .reset_index()
        )

        # Create files
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            data["timestamp"] = data["timestamp"].dt.tz_localize(None)
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path.joinpath(file_name), file_name)

    def export_generations(
        self, project_slug: str, username: str, params: ExportGenerationsParams
    ) -> DataFrame:
        # get the elements
        table = self.generations.get_generated(
            project_slug=project_slug,
            user_name=username,
        )

        # apply filters on the generated
        table["answer"] = self.generations.filter(table["answer"], params.filters)

        # join the text
        table = table.join(self.content["text"], on="index")

        return table

    def get_active_users(self, period: int = 300):
        """
        Get current active users on the time period
        """
        users = self.db_manager.users_service.get_distinct_users(self.name, period)
        return users

    def get_process(self, kind: str | list, user: str):
        """
        Get current processes
        """
        if isinstance(kind, str):
            kind = [kind]
        return [e for e in self.computing if e.user == user and e.kind in kind]

    def export_raw(self, project_slug: str):
        """
        Export raw data
        To be able to export, need to copy in the static folder
        """
        name = f"{project_slug}_data_all.parquet"
        target_dir = self.params.dir if self.params.dir is not None else Path(".")
        path_origin = target_dir.joinpath("data_all.parquet")
        folder_target = f"{config.data_path}/projects/static/{project_slug}"
        if not Path(folder_target).exists():
            os.makedirs(folder_target)
        path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
        if not Path(path_target).exists():
            shutil.copyfile(path_origin, path_target)
        return StaticFileModel(name=name, path=f"{project_slug}/{name}")

    def update_project(self, update: ProjectUpdateModel) -> None:
        """
        Update project parameters

        For text/contexts/expand, it needs to draw from raw data
        """

        if not self.params.dir:
            raise ValueError("No directory for project")

        # flag if needed to drop features
        drop_features = False
        df = None

        # update the name
        if update.project_name and update.project_name != self.params.project_name:
            self.params.project_name = update.project_name

        # update the language
        if update.language and update.language != self.params.language:
            self.params.language = update.language
            drop_features = True

        # update the context columns by modifying the train data
        if update.cols_context and set(update.cols_context) != set(self.params.cols_context):
            if df is None:
                df = pd.read_parquet(
                    self.params.dir.joinpath("data_all.parquet"),
                    columns=update.cols_context,
                )
            self.content.drop(columns=self.params.cols_context, inplace=True)
            self.content = pd.concat([self.content, df.loc[self.content.index]], axis=1)
            self.content.to_parquet(self.params.dir.joinpath("train.parquet"))
            self.params.cols_context = update.cols_context
            print("Context updated")

        # update the text columns
        if update.cols_text and set(update.cols_text) != set(self.params.cols_text):
            if df is None:
                df = pd.read_parquet(
                    self.params.dir.joinpath("data_all.parquet"),
                    columns=update.cols_text,
                )
            df_sub = df.loc[self.content.index]
            df_sub["text"] = df_sub[update.cols_text].apply(
                lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
            )
            self.content["text"] = df_sub["text"]
            self.content.to_parquet(self.params.dir.joinpath("train.parquet"))
            self.params.cols_text = update.cols_text
            drop_features = True
            del df_sub
            print("Texts updated")

        # update the train set
        if update.add_n_train and update.add_n_train > 0:
            if df is None:
                df = pd.read_parquet(
                    self.params.dir.joinpath("data_all.parquet"),
                    columns=list(self.content.columns),
                )
            # index of elements used
            elements_index = list(self.content.index)
            if self.schemes.test is not None:
                elements_index += list(self.schemes.test.index)

            # take elements that are not in index
            df = df[~df.index.isin(elements_index)]

            # sample elements
            elements_to_add = df.sample(update.add_n_train)

            # drop na elements to avoid problems
            elements_to_add = elements_to_add[elements_to_add["text"].notna()]

            self.content = pd.concat([self.content, elements_to_add])
            self.content.to_parquet(self.params.dir.joinpath("train.parquet"))

            # update params
            self.params.n_train = len(self.content)

            # drop existing features
            drop_features = True

            # restart the project
            del elements_to_add
            print("Train set updated")

        if df is not None:
            del df

        if drop_features:
            self.drop_features()

        # update the database
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )
        return None

    def drop_features(self) -> None:
        """
        Clean all the features of the project
        """
        if not self.params.dir:
            raise ValueError("No directory for project")
        self.content[[]].to_parquet(self.params.dir.joinpath("features.parquet"), index=True)
        self.features.projects_service.delete_all_features(self.name)

    def start_languagemodel_training(self, bert: BertModelModel, username: str) -> None:
        """
        Launch a training process
        """
        # Check if there is no other competing processes : 1 active process by user
        if len(self.languagemodels.current_user_processes(username)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )
        # get data
        df = self.schemes.get_scheme_data(bert.scheme, complete=True)
        df = df[["text", "labels"]].dropna()

        # management for multilabels / dichotomize
        if bert.dichotomize is not None:
            df["labels"] = df["labels"].apply(
                lambda x: self.schemes.dichotomize(x, bert.dichotomize)
            )
            bert.name = f"{bert.name}_multilabel_on_{bert.dichotomize}"

        # remove class under the threshold
        label_counts = df["labels"].value_counts()
        df = df[df["labels"].isin(label_counts[label_counts >= bert.class_min_freq].index)]

        # remove class requested by the user
        if len(bert.exclude_labels) > 0:
            df = df[~df["labels"].isin(bert.exclude_labels)]
            bert.name = f"{bert.name}_exclude_labels_"

        # balance the dataset based on the min class
        if bert.class_balance:
            min_freq = df["labels"].value_counts().sort_values().min()
            df = (
                df.groupby("labels")
                .apply(lambda x: x.sample(min_freq))
                .reset_index(level=0, drop=True)
            )

        # launch training process
        self.languagemodels.start_training_process(
            name=bert.name,
            project=self.name,
            user=username,
            scheme=bert.scheme,
            df=df,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=bert.base_model,
            params=bert.params,
            test_size=bert.test_size,
        )

    def start_generation(self, request: GenerationRequest, username: str) -> None:
        """
        Start a generation process
        """
        extract = self.schemes.get_sample(request.scheme, request.n_batch, request.mode)
        if len(extract) == 0:
            raise Exception("No elements available for generation")
        model = self.generations.generations_service.get_gen_model(request.model_id)
        # add task to the queue
        unique_id = self.queue.add_task(
            "generation",
            self.name,
            GenerateCall(
                path_process=self.params.dir,
                username=username,
                project_slug=self.name,
                df=extract,
                prompt=request.prompt,
                model=GenerationModel(**model.__dict__),
            ),
        )
        self.computing.append(
            GenerationComputing(
                unique_id=unique_id,
                user=username,
                project=self.name,
                model_id=request.model_id,
                number=request.n_batch,
                time=datetime.now(),
                kind="generation",
                get_progress=GenerateCall.get_progress_callback(
                    self.params.dir.joinpath(unique_id) if self.params.dir is not None else None
                ),
            )
        )

    def update_processes(self) -> None:
        """
        Update completed processes and do specific operations regarding their kind
        - get the result from the queue
        - add the result if needed
        - manage error if needed

        # TODO : REFACTOR THIS FUNCTION

        """
        add_predictions = {}

        # loop on the current process
        for e in self.computing.copy():
            process = self.queue.get(e.unique_id)

            # case of not in queue
            if process is None:
                logging.warning("Problem : id in computing not in queue")
                self.computing.remove(e)
                continue

            # check if the process is done, else continue
            if process["future"] is None or not process["future"].done():
                continue

            # get the future
            future = process["future"]

            # manage different tasks

            # case for bert fine-tuning
            if e.kind == "train_bert":
                model = cast(LMComputing, e)
                try:
                    print(process)
                    error = future.exception()
                    if error:
                        raise Exception(str(error))
                    self.languagemodels.add(model)
                    print("Bert training achieved")
                    logging.debug("Bert training achieved")
                except Exception as ex:
                    self.db_manager.language_models_service.delete_model(
                        self.name, model.model_name
                    )
                    self.errors.append(
                        [
                            datetime.now(TIMEZONE),
                            "Error in bert model training",
                            str(ex),
                        ]
                    )
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for bertmodel prediction
            if e.kind == "predict_bert":
                prediction = cast(LMComputing, e)
                try:
                    error = future.exception()
                    if error:
                        raise Exception(str(error))
                    results = future.result()

                    # case of predict_train : transform to feature
                    if (
                        results is not None
                        and results.path
                        and "predict_train.parquet" in results.path
                    ):
                        add_predictions["predict_" + prediction.model_name] = results.path
                    self.languagemodels.add(prediction)
                    print("Bert predicting achieved")
                    logging.debug("Bert predicting achieved")
                except Exception as ex:
                    self.errors.append(
                        [
                            datetime.now(TIMEZONE),
                            "Error in model predicting",
                            str(ex),
                        ]
                    )
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for simplemodels
            if e.kind == "simplemodel":
                sm = cast(SimpleModelComputing, e)
                try:
                    error = future.exception()
                    if error:
                        raise Exception(str(error))
                    results = future.result()
                    self.simplemodels.add(sm, results)
                    print("Simplemodel trained")
                    logging.debug("Simplemodel trained")
                except Exception as ex:
                    self.errors.append([datetime.now(TIMEZONE), "simplemodel failed", str(ex)])
                    logging.error("Simplemodel failed", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for features
            if e.kind == "feature":
                feature_computation = cast(FeatureComputing, e)
                try:
                    error = future.exception()
                    if error:
                        raise Exception("from task" + str(error))
                    results = future.result()
                    self.features.add(
                        feature_computation.name,
                        feature_computation.type,
                        feature_computation.user,
                        feature_computation.parameters,
                        results,
                    )
                    print("Feature added", feature_computation.name)
                except Exception as ex:
                    self.errors.append(
                        [datetime.now(TIMEZONE), "Error in feature processing", str(ex)]
                    )
                    print("Error in feature processing", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for projections
            if e.kind == "projection":
                projection = cast(ProjectionComputing, e)
                try:
                    results = future.result()
                    self.projections.add(projection, results)
                    print("Projection added", projection.name)
                    logging.debug("projection added")
                except Exception as ex:
                    self.errors.append(
                        [
                            datetime.now(TIMEZONE),
                            "Error in feature vizualisation queue",
                            str(ex),
                        ]
                    )
                    logging.error("Error in feature vizualisation queue", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for generations
            if e.kind == "generation":
                try:
                    results = future.result()
                    r = cast(
                        list[GenerationResult],
                        results,
                    )
                    for row in r:
                        self.generations.add(
                            user=row.user,
                            project_slug=row.project_slug,
                            element_id=row.element_id,
                            model_id=row.model_id,
                            prompt=row.prompt,
                            answer=row.answer,
                        )
                except Exception as ex:
                    self.errors.append(
                        [
                            datetime.now(TIMEZONE),
                            "Error in generation queue",
                            getattr(ex, "message", repr(ex)),
                        ]
                    )
                    logging.warning("Error in generation queue", getattr(ex, "message", repr(ex)))
                    print("Error in generation queue", getattr(ex, "message", repr(ex)))
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

        # if there are predictions, add them
        for f in add_predictions:
            try:
                # load the prediction probabilities minus one
                df = pd.read_parquet(add_predictions[f])
                df = df.drop(columns=["entropy", "prediction"])
                df = df[df.columns[0:-1]]
                name = f.replace("__", "_")  # avoid __ in the name for features
                # if the feature already exists, delete it first
                if self.features.exists(name):
                    self.features.delete(name)
                # add it
                self.features.add(
                    name=name,
                    kind="prediction",
                    parameters={},
                    username="system",
                    new_content=df,
                )
                logging.debug("Add feature" + str(name))
            except Exception as ex:
                self.errors.append(
                    [
                        datetime.now(TIMEZONE),
                        "Error in adding prediction",
                        str(ex),
                    ]
                )
                logging.error("Error in addind prediction", ex)

        # clean errors older than 15 minutes
        delta = datetime.now(TIMEZONE) - timedelta(minutes=15)
        self.errors = [error for error in self.errors if error[0] >= delta]

        return None
