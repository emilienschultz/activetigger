import io
import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import]
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse
from pandas import DataFrame

from activetigger.bertopic import Bertopic
from activetigger.config import config
from activetigger.data import Data
from activetigger.datamodels import (
    BertModelModel,
    BertopicComputing,
    ElementInModel,
    ElementOutModel,
    EvalSetDataModel,
    ExportGenerationsParams,
    FeatureComputing,
    GenerationComputing,
    GenerationModel,
    GenerationRequest,
    GenerationResult,
    LMComputing,
    NextInModel,
    NextProjectStateModel,
    PredictedLabel,
    ProjectBaseModel,
    ProjectCreatingModel,
    ProjectDescriptionModel,
    ProjectionComputing,
    ProjectionOutModel,
    ProjectModel,
    ProjectStateModel,
    ProjectUpdateModel,
    QuickModelComputing,
    QuickModelInModel,
    StaticFileModel,
)
from activetigger.db.manager import DatabaseManager
from activetigger.features import Features
from activetigger.functions import clean_regex, get_dir_size, slugify
from activetigger.generation.generations import Generations
from activetigger.languagemodels import LanguageModels
from activetigger.projections import Projections
from activetigger.queue import Queue
from activetigger.quickmodels import QuickModels
from activetigger.schemes import Schemes
from activetigger.tasks.create_project import CreateProject
from activetigger.tasks.generate_call import GenerateCall
from activetigger.users import Users


class Project:
    """
    Project object
    """

    status: str
    starting_time: float
    name: str
    queue: Queue
    computing: list
    path_models: Path
    data: Data
    db_manager: DatabaseManager
    params: ProjectModel
    train: DataFrame | None
    valid: DataFrame | None
    test: DataFrame | None
    schemes: Schemes
    features: Features
    languagemodels: LanguageModels
    quickmodels: QuickModels
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
        self.status = "initial"
        self.starting_time = time.time()
        self.queue = queue
        self.computing = []
        self.db_manager = db_manager
        self.path_models = path_models
        self.name = project_slug
        self.project_slug = project_slug
        self.errors = []  # TODO Move to specific class / db in the future
        self.users = Users(self.db_manager)

        # load the project if exist
        if self.exists():
            self.status = "created"
            self.load_project(project_slug)

    def __del__(self):
        print(f"Project {self.name} deleted")
        pass

    def exists(self) -> bool:
        """
        Check if the project exists
        """
        if self.db_manager.projects_service.get_project(self.project_slug):
            return True
        return False

    def start_project_creation(self, params: ProjectBaseModel, username: str, path: Path) -> None:
        """
        Manage process creation, sending the heavy process to the queue
        """
        self.status = "creating"
        # test if the name of the column is specified
        if params.col_id is None or params.col_id == "":
            raise Exception("No column selected for the id")
        if params.cols_text is None or len(params.cols_text) == 0:
            raise Exception("No column selected for the text")
        # add the dedicated directory
        params.dir = path.joinpath(self.project_slug)

        print("Adding project to the queue")
        # send the process to the queue
        unique_id = self.queue.add_task(
            "create_project",
            self.project_slug,
            CreateProject(
                self.project_slug,
                params,
                username,
                data_all=config.data_all,
                train_file=config.train_file,
                valid_file=config.valid_file,
                test_file=config.test_file,
                features_file=config.features_file,
            ),
            queue="cpu",
        )
        # Update the register
        self.computing.append(
            ProjectCreatingModel(
                username=username,
                project_slug=self.project_slug,
                unique_id=unique_id,
                time=datetime.now(),
                kind="create_project",
                status="training",
            )
        )

    def finish_project_creation(
        self,
        username: str,
        project: ProjectModel,
        import_trainset_labels: pd.DataFrame | None = None,
        import_validset_labels: pd.DataFrame | None = None,
        import_testset_labels: pd.DataFrame | None = None,
    ) -> None:
        """
        Get the result of the queue and finish the creation process
        """
        # add the project to the database
        self.db_manager.projects_service.add_project(
            project.project_slug, jsonable_encoder(project), username
        )
        # add the default scheme (basic name or with a random number if the name already exists)
        if import_trainset_labels is None or len(import_trainset_labels.columns) == 0:
            self.db_manager.projects_service.add_scheme(
                self.project_slug, config.default_scheme, [], "multiclass", "system"
            )
        # if labels/schemes to import, add them to the database
        else:
            for col in import_trainset_labels.columns:
                scheme_name = col.replace("dataset_", "")
                delimiters = import_trainset_labels[col].str.contains("|", regex=False).sum()
                if delimiters < 5:
                    scheme_type = "multiclass"
                    scheme_labels = list(import_trainset_labels[col].dropna().unique())
                else:
                    scheme_type = "multilabel"
                    scheme_labels = list(
                        import_trainset_labels[col].dropna().str.split("|").explode().unique()
                    )
                self.db_manager.projects_service.add_scheme(
                    self.project_slug,
                    scheme_name,
                    scheme_labels,
                    scheme_type,
                    "system",
                )
                elements = [
                    {"element_id": element_id, "annotation": label, "comment": ""}
                    for element_id, label in import_trainset_labels[col].dropna().items()
                ]
                self.db_manager.projects_service.add_annotations(
                    dataset="train",
                    user_name=username,
                    project_slug=self.project_slug,
                    scheme=scheme_name,
                    elements=elements,
                )
                if import_validset_labels is not None and col in import_validset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_validset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="valid",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )
                if import_testset_labels is not None and col in import_testset_labels.columns:
                    elements = [
                        {"element_id": element_id, "annotation": label, "comment": ""}
                        for element_id, label in import_testset_labels[col].dropna().items()
                    ]
                    self.db_manager.projects_service.add_annotations(
                        dataset="test",
                        user_name=username,
                        project_slug=self.project_slug,
                        scheme=scheme_name,
                        elements=elements,
                    )

        # create user authorizations
        self.users.set_auth(username, project.project_slug, "manager")
        self.users.set_auth("root", project.project_slug, "manager")
        self.status = "created"

    def delete(self) -> None:
        """
        Delete completely a project
        """

        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # remove from database
        try:
            print("Deleting project from the database")
            self.db_manager.projects_service.delete_project(self.params.project_slug)
        except Exception as e:
            print(f"Problem with the database: {e}")
            raise ValueError("Problem with the database " + str(e))

        # remove folder of the project
        try:
            shutil.rmtree(self.params.dir)
        except Exception as e:
            raise ValueError("No directory to delete " + str(e))

        # remove static files
        if Path(f"{config.data_path}/projects/static/{self.name}").exists():
            shutil.rmtree(f"{config.data_path}/projects/static/{self.name}")

    def load_project(self, project_slug: str) -> None:
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

        # Tabular data management
        self.data = Data(
            self.params.dir,
            self.params.dir.joinpath(config.train_file),
            self.params.dir.joinpath(config.valid_file) if self.params.valid else None,
            self.params.dir.joinpath(config.test_file) if self.params.test else None,
        )

        # create specific management objets
        self.schemes = Schemes(
            project_slug,
            self.db_manager,
            self.data,
        )
        self.features = Features(
            project_slug,
            self.params.dir.joinpath(config.features_file),
            self.params.dir.joinpath(config.data_all),
            self.params.dir.joinpath(config.train_file),
            self.params.dir.joinpath(config.valid_file),
            self.params.dir.joinpath(config.test_file),
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
            config.file_models,
        )
        self.quickmodels = QuickModels(
            project_slug, self.params.dir, self.queue, self.computing, self.db_manager
        )
        self.generations = Generations(
            self.db_manager, cast(list[GenerationComputing], self.computing)
        )
        self.projections = Projections(self.params.dir, self.computing, self.queue)
        self.bertopic = Bertopic(
            project_slug,
            self.params.dir,
            self.queue,
            self.computing,
            self.features,
            self.db_manager,
        )

    def load_params(self, project_slug: str) -> ProjectModel:
        """
        Load params from database
        """
        existing_project = self.db_manager.projects_service.get_project(project_slug)
        if existing_project:
            return ProjectModel(**existing_project["parameters"])
        else:
            raise NameError(f"{project_slug} does not exist.")

    def drop_evalset(self, dataset: str) -> None:
        """
        Clean all the test data of the project
        - remove the file
        - remove all the annotations in the database
        - set the flag to False
        """
        if not self.params.dir:
            raise Exception("No directory for project")
        path = self.params.dir.joinpath(f"{dataset}.parquet")
        if not path.exists():
            raise Exception("No eval data available")
        os.remove(path)
        self.db_manager.projects_service.delete_annotations_evalset(
            self.params.project_slug, dataset
        )
        if dataset == "test":
            self.data.test = None
            self.params.test = False
        if dataset == "valid":
            self.data.valid = None
            self.params.valid = False
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

        # reset the features file
        self.features.reset_features_file()

    def add_evalset(
        self, dataset, evalset: EvalSetDataModel, username: str, project_slug: str
    ) -> None:
        """
        Add a eval dataset (test or valid)

        The eval dataset should :
        - not contains NA
        - have a unique id different from the complete dataset

        The id will be modified to indicate imported

        """
        if len(evalset.cols_text) == 0:
            raise Exception("No text column selected for the evalset")

        if self.params.dir is None:
            raise Exception("Cannot add eval data without a valid dir")

        if evalset.col_label == "":
            evalset.col_label = None

        if dataset not in ["test", "valid"]:
            raise Exception("Dataset should be test or valid")

        if dataset == "test" and self.params.test:
            raise Exception("There is already a test dataset")

        if dataset == "valid" and self.params.valid:
            raise Exception("There is already a valid dataset")

        csv_buffer = io.StringIO(evalset.csv)
        df = pd.read_csv(
            csv_buffer,
            dtype={evalset.col_id: str, **{col: str for col in evalset.cols_text}},
            nrows=evalset.n_eval,
        )

        if len(df) > 10000:
            raise Exception("You valid set is too large")

        # create text column
        df["text"] = df[evalset.cols_text].apply(
            lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
        )

        # change names
        if not evalset.col_label:
            df = df.rename(columns={evalset.col_id: "id"})
        else:
            df = df.rename(
                columns={
                    evalset.col_id: "id",
                    evalset.col_label: "label",
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
        if evalset.col_label and evalset.scheme:
            # Check the label columns if they match the scheme or raise error
            scheme = self.schemes.available()[evalset.scheme].labels
            for label in df["label"].dropna().unique():
                if label not in scheme:
                    raise Exception(f"Label {label} not in the scheme {evalset.scheme}")

            elements = [
                {"element_id": element_id, "annotation": label, "comment": ""}
                for element_id, label in df["label"].dropna().items()
            ]
            self.db_manager.projects_service.add_annotations(
                dataset=dataset,
                user_name=username,
                project_slug=project_slug,
                scheme=evalset.scheme,
                elements=elements,
            )
            print("Valid labels imported")

        # write the dataset
        if dataset == "test":
            df[["text"]].to_parquet(self.params.dir.joinpath(config.test_file))
            self.data.test = df[["text"]]
            self.params.test = True
        else:
            df[["text"]].to_parquet(self.params.dir.joinpath(config.valid_file))
            self.data.valid = df[["text"]]
            self.params.valid = True

        # update the database
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )

        # reset the features file
        self.features.reset_features_file()

    def retrain_quickmodel(self, name: str, scheme: str, username: str) -> None:
        """
        Retrain a quickmodel
        """

        # Get old model parameters in a QuickModelInModel
        model = self.quickmodels.get(name)
        quickmodel = QuickModelInModel(
            name=name,
            scheme=scheme,
            model=model.model_type,
            features=model.features,
            params=model.model_params,
            standardize=model.standardize,
            dichotomize=model.model_params.get("dichotomize", None),
            cv10=model.cv10,
            balance_classes=model.balance_classes
        )
        self.train_quickmodel(quickmodel, username, retrain=True)

    def train_quickmodel(
        self,
        quickmodel: QuickModelInModel,
        username: str,
        n_min_annotated: int = 3,
        retrain: bool = False,
    ) -> None:
        """
        Build all the information before calling the quickmodel computation
        retrain : if True, will delete the previous model with the same name
        """
        # Tests
        availabe_schemes = self.schemes.available()
        quickmodel.features = [i for i in quickmodel.features if i is not None]
        if quickmodel.features is None or len(quickmodel.features) == 0:
            raise Exception("No features selected")
        if quickmodel.model not in list(self.quickmodels.available_models.keys()):
            raise Exception("Model not available")
        if quickmodel.scheme not in availabe_schemes:
            raise Exception("Scheme not available")
        if len(availabe_schemes[quickmodel.scheme].labels) < 2:
            raise Exception("Not enough labels in the scheme")
        quickmodel.name = slugify(quickmodel.name)
        exist = self.quickmodels.exists(quickmodel.name)
        if exist and not retrain:
            raise Exception("A quickmodel with this name already exists")
        if not exist and retrain:
            raise Exception("No quickmodel with this name to retrain")

        # only dfm feature for multi_naivebayes (FORCE IT if available else error)
        if quickmodel.model == "multi_naivebayes":
            if "dfm" not in self.features.map:
                raise Exception("No dfm feature available")
            quickmodel.features = ["dfm"]
            quickmodel.standardize = False

        if quickmodel.params is None:
            params = None
        else:
            params = dict(quickmodel.params)
        # add information on the target of the model
        if quickmodel.dichotomize is not None and params is not None:
            params["dichotomize"] = quickmodel.dichotomize

        # get data
        df_features = self.features.get(quickmodel.features, dataset=["train"])
        df_scheme = self.schemes.get_scheme_data(scheme=quickmodel.scheme)

        # management for multilabels / dichotomize
        if quickmodel.dichotomize is not None:
            df_scheme["labels"] = df_scheme["labels"].apply(
                lambda x: self.schemes.dichotomize(x, quickmodel.dichotomize)
            )

        # test for a minimum of annotated elements
        counts = df_scheme["labels"].value_counts()
        valid_categories = counts[counts >= n_min_annotated]
        if len(valid_categories) < 2:
            raise Exception(
                f"Not enough annotated elements (should be more than {n_min_annotated})"
            )

        col_features = list(df_features.columns)
        data = pd.concat([df_scheme, df_features], axis=1)

        logger_quickmodel = logging.getLogger("quickmodel")
        logger_quickmodel.info("Building the quickmodel request")
        self.quickmodels.compute_quickmodel(
            project_slug=self.params.project_slug,
            user=username,
            scheme=quickmodel.scheme,
            features=quickmodel.features,
            name=quickmodel.name,
            model_type=quickmodel.model,
            df=data,
            col_labels="labels",
            col_features=col_features,
            model_params=params,
            standardize=quickmodel.standardize or False,
            cv10=quickmodel.cv10 or False,
            balance_classes=quickmodel.balance_classes or False,
            retrain=retrain,
            texts=self.data.train["text"] if self.data.train is not None else None,
        )

    def get_model_prediction(self, type: str, name: str) -> pd.DataFrame:
        """
        Get prediction of a model or raise an error
        - quickmodel
        - languagemodel
        """
        if type == "quickmodel":
            if not self.quickmodels.exists(name):
                raise Exception("Quickmodel doesn't exist")
            else:
                prediction = self.quickmodels.get_prediction(name)
        elif type == "languagemodel":
            if not self.languagemodels.exists(name):
                raise Exception("Languagemodel doesn't exist")
            else:
                prediction = self.languagemodels.get_prediction(name)
        else:
            raise Exception("Model type not recognized")
        return prediction

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
            if self.data.test is None:
                raise ValueError("No test dataset available")
            df = self.schemes.get_scheme_data(next.scheme, complete=True, kind=["test"])
        elif next.dataset == "valid":
            if self.data.valid is None:
                raise ValueError("No valid dataset available")
            df = self.schemes.get_scheme_data(next.scheme, complete=True, kind=["valid"])
        else:
            df = self.schemes.get_scheme_data(next.scheme, complete=True, kind=["train"])

        # the filter for the sample
        if next.sample == "untagged":
            f = df["labels"].isna()
        elif next.sample == "tagged":
            # on a specific label
            if next.label is not None and next.label in df["labels"].unique():
                f = df["labels"] == next.label
            else:
                f = df["labels"].notna()
            # for a specific user if specified
            if next.user is not None and next.user != "":
                f_user = df["user"] == next.user
                f = f & f_user
        else:
            f = df["labels"].apply(lambda x: True)

        # add a regex condition to the selection
        if next.filter:
            # sanitize
            df["ID"] = df.index  # duplicate the id column
            filter_san = clean_regex(next.filter)
            if "CONTEXT=" in filter_san:  # case to search in the context
                f_regex = (
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
                f_regex = cast(
                    pd.Series, df[self.params.cols_context].eval(filter_san.replace("QUERY=", ""))
                )
            else:
                f_regex = df["text"].str.contains(filter_san, regex=True, case=True, na=False)
            f = f & f_regex

        # manage frame selection (if projection, only in the box)
        if next.frame and len(next.frame) == 4:
            if username in self.projections.available:
                if self.projections.available[username].data is not None:
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

        # check conditions for active learning and get proba
        if next.selection in ["maxprob", "active"]:
            if next.model_active is None:
                raise Exception("Model active is required")
            prediction = self.get_model_prediction(next.model_active.type, next.model_active.value)
            proba = prediction.reindex(f.index)

        # higher prob for the label_maxprob, only possible if the model has been trained
        if next.selection == "maxprob":
            if next.label_maxprob is None:  # default label to first
                raise Exception("Label maxprob is required")
            # use the history to not send already tagged data
            ss_maxprob = (
                proba[f][next.label_maxprob]
                .drop(next.history, errors="ignore")
                .sort_values(ascending=False)
            )  # get max proba id
            element_id = ss_maxprob.index[0]
            n_sample = f.sum()
            indicator = f"probability: {round(proba.loc[element_id, next.label_maxprob], 2)}"

        # higher entropy, only possible if the model has been trained
        if next.selection == "active":
            # use the history to not send already tagged data
            ss_active = (
                proba[f]["entropy"].drop(next.history, errors="ignore").sort_values(ascending=False)
            )  # get max entropy id
            element_id = ss_active.index[0]
            n_sample = f.sum()
            indicator = round(proba.loc[element_id, "entropy"], 2)
            indicator = f"entropy: {indicator}"

        # get prediction of the id if it exists
        predict = PredictedLabel(label=None, proba=None, entropy=None)

        if (
            next.model_active is not None
            and self.quickmodels.exists(next.model_active.value)
            and next.dataset == "train"
        ):
            predict = self.quickmodels.get_prediction_element(next.model_active.value, element_id)

        # get all tags already existing for the element
        previous = self.schemes.projects_service.get_annotations_by_element(
            self.params.project_slug, next.scheme, element_id
        )

        if next.dataset in ["test", "valid"]:
            limit = 1200
            context = {}
        else:
            if self.data.train is None:
                raise Exception("Train dataset is not defined")
            limit = int(self.data.train.loc[element_id, "limit"])
            # get context
            context = dict(
                self.data.train.fillna("NA").loc[element_id, self.params.cols_context].apply(str)
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
        element: ElementInModel,
        user: str | None = None,
    ) -> ElementOutModel:
        """
        Get an element of the database
        Separate train/test dataset

        TODO : get next and get element could be merged
        """
        history = None

        if element.dataset == "valid" and self.data.valid is not None:
            if element.element_id not in self.data.valid.index:
                raise Exception("Element does not exist.")
            if element.scheme is not None:
                history = self.schemes.projects_service.get_annotations_by_element(
                    self.params.project_slug, element.scheme, element.element_id
                )
            return ElementOutModel(
                element_id=element.element_id,
                text=str(self.data.valid.loc[element.element_id, "text"]),
                context={},
                selection="valid",
                info="",
                predict=PredictedLabel(label=None, proba=None, entropy=None),
                frame=None,
                limit=1200,
                history=history,
            )

        if element.dataset == "test" and self.data.test is not None:
            if element.element_id not in self.data.test.index:
                raise Exception("Element does not exist.")
            if element.scheme is not None:
                history = self.schemes.projects_service.get_annotations_by_element(
                    self.params.project_slug, element.scheme, element.element_id
                )
            return ElementOutModel(
                element_id=element.element_id,
                text=str(self.data.test.loc[element.element_id, "text"]),
                context={},
                selection="test",
                info="",
                predict=PredictedLabel(label=None, proba=None, entropy=None),
                frame=None,
                limit=1200,
                history=history,
            )

        if element.dataset == "train":
            if self.data.train is None:
                raise Exception("Train dataset is not defined")
            if element.element_id not in self.data.train.index:
                raise Exception("Element does not exist.")

            # get prediction if it exists
            predict = PredictedLabel(label=None, proba=None, entropy=None)
            if element.active_model is not None:
                predict = self.quickmodels.get_prediction_element(
                    element.active_model.value, element.element_id
                )

            # get element tags
            if element.scheme is not None:
                history = self.schemes.projects_service.get_annotations_by_element(
                    self.params.project_slug, element.scheme, element.element_id
                )

            context = cast(
                dict[str, Any],
                self.data.train.loc[element.element_id, self.params.cols_context]  # type: ignore[index]
                .fillna("NA")
                .astype(str)
                .to_dict(),
            )

            return ElementOutModel(
                element_id=element.element_id,
                text=str(self.data.train.loc[element.element_id, "text"]),
                context=context,
                selection="request",
                predict=predict,
                info="get specific",
                frame=None,
                limit=cast(int, self.data.train.loc[element.element_id, "limit"]),
                history=history,
            )

        raise Exception("Dataset does not exist.")

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params

    @staticmethod
    def compute_annotations_distribution(df: DataFrame, kind: str) -> dict[str, int]:
        if kind == "multiclass":
            return json.loads(df["labels"].value_counts().to_json())
        elif kind == "multilabel":
            return json.loads(df["labels"].str.split("|").explode().value_counts().to_json())
        elif kind == "span":
            r = (
                df["labels"]
                .apply(lambda x: json.loads(x) if pd.notna(x) else [])
                .explode()
                .apply(lambda x: x["tag"] if isinstance(x, dict) and "tag" in x else None)
            )
            return json.loads(r.value_counts().to_json())
        else:
            raise Exception("Not implemented for this kind of scheme")

    def get_statistics(self, scheme: str | None) -> ProjectDescriptionModel:
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
        users = self.db_manager.users_service.get_coding_users(scheme, self.params.project_slug)
        df_train = self.schemes.get_scheme_data(scheme, kind=["train"])
        train_annotated_distribution = self.compute_annotations_distribution(df_train, kind)

        # part valid
        if self.params.valid and (self.data.valid is not None):
            df_valid = self.schemes.get_scheme_data(scheme, kind=["valid"])
            valid_set_n = len(self.data.valid)
            valid_annotated_n = len(df_valid.dropna(subset=["labels"]))
            valid_annotated_distribution = self.compute_annotations_distribution(df_valid, kind)
        else:
            valid_set_n = None
            valid_annotated_n = None
            valid_annotated_distribution = None

        # part test
        if self.params.test and (self.data.test is not None):
            df_test = self.schemes.get_scheme_data(scheme, kind=["test"])
            test_set_n = len(self.data.test)
            test_annotated_n = len(df_test.dropna(subset=["labels"]))
            test_annotated_distribution = self.compute_annotations_distribution(df_test, kind)
        else:
            test_set_n = None
            test_annotated_n = None
            test_annotated_distribution = None

        return ProjectDescriptionModel(
            users=users,
            train_set_n=len(self.data.train) if self.data.train is not None else 0,
            train_annotated_n=len(df_train.dropna(subset=["labels"])),
            train_annotated_distribution=train_annotated_distribution,
            valid_set_n=valid_set_n,
            valid_annotated_n=valid_annotated_n,
            valid_annotated_distribution=valid_annotated_distribution,
            test_set_n=test_set_n,
            test_annotated_n=test_annotated_n,
            test_annotated_distribution=test_annotated_distribution,
            sm_10cv=None,
        )

    def get_projection(
        self, username: str, scheme: str, model: str | None = None
    ) -> ProjectionOutModel | None:
        """
        Get projection if computed
        """
        projection = self.projections.get(username)
        if projection is None:
            return None
        # get annotations
        df = self.schemes.get_scheme_data(scheme, complete=True, kind=["train"])
        data = projection.data
        data["labels"] = df["labels"].fillna("NA")

        # get & add predictions if available
        if model and self.quickmodels.exists(model):
            data["prediction"] = self.quickmodels.get_prediction(model)["prediction"]

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
        Collecting states for submodules
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
            quickmodel=self.quickmodels.state(),
            languagemodels=self.languagemodels.state(),
            projections=self.projections.state(),
            generations=self.generations.state(),
            bertopic=self.bertopic.state(),
            errors=self.errors,
            memory=get_dir_size(str(self.params.dir)),
            last_activity=self.db_manager.logs_service.get_last_activity_project(
                self.params.project_slug
            ),
            users=self.users.state(self.params.project_slug),
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

        data = self.features.get(features, dataset="annotable")

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path.joinpath(file_name))
        if format == "parquet":
            data.to_parquet(path.joinpath(file_name))
        if format == "xlsx":
            data.to_excel(path.joinpath(file_name))

        return FileResponse(path=path.joinpath(file_name), filename=file_name)

    def export_data(
        self, scheme: str, dataset: str = "train", format: str = "parquet", dropna: bool = True
    ) -> FileResponse:
        """
        Export annotation data in different formats
        - for a scheme
        - for all schemes
        """
        path = self.params.dir  # path of the data
        if path is None:
            raise ValueError("Problem of filesystem for project")

        # test or train
        if dataset == "test":
            if not self.params.test:
                raise Exception("No test data available")
            if scheme == "all":
                schemes = self.schemes.available()
                if len(schemes) == 0:
                    raise Exception("No scheme available")
                data = pd.concat(
                    {
                        s: self.schemes.get_scheme_data(s, complete=True, kind=["test"])["labels"]
                        for s in schemes
                    },
                    axis=1,
                )
                file_name = f"data_test_{self.name}_all_schemes.{format}"
                dropna = False
            else:
                data = self.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["test"])
                file_name = f"data_test_{self.name}_{scheme}.{format}"
        else:
            if scheme == "all":
                schemes = self.schemes.available()
                if len(schemes) == 0:
                    raise Exception("No scheme available")
                data = pd.concat(
                    {
                        s: self.schemes.get_scheme_data(s, complete=True, kind=["train"])["labels"]
                        for s in schemes
                    },
                    axis=1,
                )
                file_name = f"data_train_{self.name}_all_schemes.{format}"
                dropna = False
            else:
                data = self.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["train"])
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

        return FileResponse(path.joinpath(file_name), filename=file_name)

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
        if self.data.train is None:
            raise Exception("No train data available")
        table = table.join(self.data.train["text"], on="index")

        return table

    def get_process(
        self, kind: str | list, user: str
    ) -> list[FeatureComputing | LMComputing | QuickModelComputing]:
        """
        Get current processes
        """
        if isinstance(kind, str):
            kind = [kind]
        return [e for e in self.computing if e.user == user and e.kind in kind]

    def export_raw(self, project_slug: str) -> StaticFileModel:
        """
        Export raw data
        To be able to export, need to copy in the static folder
        """
        target_dir = self.params.dir if self.params.dir is not None else Path(".")
        path_origin = target_dir.joinpath("data_all.parquet")
        folder_target = f"{config.data_path}/projects/static/{project_slug}"
        if not Path(folder_target).exists():
            os.makedirs(folder_target)
        files = [i for i in os.listdir(folder_target) if "_data_all_" in i]
        # file already exists
        if len(files) > 0:
            name = files[0]
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
        # create the file with a unique id
        else:
            name = f"{project_slug}_data_all_{uuid.uuid4()}.parquet"
            path_target = f"{config.data_path}/projects/static/{project_slug}/{name}"
            shutil.copyfile(path_origin, path_target)
        return StaticFileModel(name=name, path=f"{project_slug}/{name}")

    def update_project(self, update: ProjectUpdateModel) -> None:
        """
        Update project parameters

        For text/contexts/expand, it needs to draw from raw data
        """

        if not self.params.dir:
            raise ValueError("No directory for project")
        if self.data.train is None:
            raise ValueError("No train data for project")

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
            self.data.train.drop(columns=self.params.cols_context, inplace=True)
            self.data.train = pd.concat([self.data.train, df.loc[self.data.train.index]], axis=1)
            self.data.train.to_parquet(self.params.dir.joinpath(config.train_file))
            self.params.cols_context = update.cols_context
            print("Context updated")

        # update the text columns
        if update.cols_text and set(update.cols_text) != set(self.params.cols_text):
            if df is None:
                df = pd.read_parquet(
                    self.params.dir.joinpath("data_all.parquet"),
                    columns=update.cols_text,
                )
            df_sub = df.loc[self.data.train.index]
            df_sub["text"] = df_sub[update.cols_text].apply(
                lambda x: "\n\n".join([str(i) for i in x if pd.notnull(i)]), axis=1
            )
            self.data.train["text"] = df_sub["text"]
            self.data.train.to_parquet(self.params.dir.joinpath(config.train_file))
            self.params.cols_text = update.cols_text
            drop_features = True
            del df_sub
            print("Texts updated")

        # update the train set
        if update.add_n_train and update.add_n_train > 0:
            if df is None:
                df = pd.read_parquet(
                    self.params.dir.joinpath("data_all.parquet"),
                    columns=list(self.data.train.columns),
                )
            # index of elements used
            elements_index = list(self.data.train.index)
            if self.data.test is not None:
                elements_index += list(self.data.test.index)
            if self.data.valid is not None:
                elements_index += list(self.data.valid.index)

            # take elements that are not in index
            df = df[~df.index.isin(elements_index)]

            # sample elements
            elements_to_add = df.sample(update.add_n_train)

            # drop na elements to avoid problems
            elements_to_add = elements_to_add[elements_to_add["text"].notna()]

            self.data.train = pd.concat([self.data.train, elements_to_add])
            self.data.train.to_parquet(self.params.dir.joinpath(config.train_file))

            # update params
            self.params.n_train = len(self.data.train)

            # drop existing features
            drop_features = True

            # restart the project
            del elements_to_add
            print("Train set updated")

        if df is not None:
            del df

        if drop_features:
            self.features.reset_features_file()

        # update the database
        self.db_manager.projects_service.update_project(
            self.params.project_slug, jsonable_encoder(self.params)
        )
        return None

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
        df = self.schemes.get_scheme_data(bert.scheme, kind=["train"], complete=True)
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
            loss=bert.loss,
            max_length=bert.max_length,
            auto_max_length=bert.auto_max_length,
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
                cols_context=self.params.cols_context,
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
            if process.future is None or not process.future.done():
                continue

            # get the future for process done
            future = process.future
            exception = future.exception()
            if exception:
                print(f"Error in {e.kind} : {exception}")
                self.errors.append(
                    [datetime.now(config.timezone), f"Error for process {e.kind}", str(exception)]
                )
                self.computing.remove(e)
                self.queue.delete(e.unique_id)

                # specific case for project creation
                if e.kind == "create_project":
                    self.status = "error"
                continue

            # case for project creation
            if e.kind == "create_project":
                e = cast(ProjectCreatingModel, e)
                try:
                    results = future.result()
                    if results is None:
                        print("No result from project creation")
                        raise Exception("No result from project creation")
                    self.finish_project_creation(
                        e.username, results[0], results[1], results[2], results[3]
                    )
                except Exception as ex:
                    self.errors.append(
                        [datetime.now(config.timezone), "Error in project creation", str(ex)]
                    )
                    logging.error("Error in project creation", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for bert fine-tuning
            if e.kind == "train_bert":
                model = cast(LMComputing, e)
                try:
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
                            datetime.now(config.timezone),
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

                    # case of predict on annotable : transform to feature
                    if (
                        results is not None
                        and results.path
                        and "predict_annotable.parquet" in results.path
                    ):
                        add_predictions["predict_" + prediction.model_name] = results.path
                    self.languagemodels.add(prediction)
                    print("Bert predicting achieved")
                    logging.debug("Bert predicting achieved")
                except Exception as ex:
                    self.errors.append(
                        [
                            datetime.now(config.timezone),
                            "Error in model predicting",
                            str(ex),
                        ]
                    )
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for quickmodels training
            if e.kind == "train_quickmodel":
                sm = cast(QuickModelComputing, e)
                try:
                    error = future.exception()
                    if error:
                        raise Exception(str(error))
                    self.quickmodels.add(sm)
                    print("Quickmodel trained")
                    logging.debug("Quickmodel trained")
                except Exception as ex:
                    self.errors.append(
                        [datetime.now(config.timezone), "quickmodel failed", str(ex)]
                    )
                    logging.error("Quickmodel failed", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for quickmodel prediction
            if e.kind == "predict_quickmodel":
                sm = cast(QuickModelComputing, e)
                try:
                    error = future.exception()
                    if error:
                        raise Exception(str(error))
                except Exception as ex:
                    self.errors.append(
                        [datetime.now(config.timezone), "quickmodel failed", str(ex)]
                    )
                    logging.error("Quickmodel failed", ex)
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
                        [datetime.now(config.timezone), "Error in feature processing", str(ex)]
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
                            datetime.now(config.timezone),
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
                            datetime.now(config.timezone),
                            "Error in generation queue",
                            getattr(ex, "message", repr(ex)),
                        ]
                    )
                    logging.warning("Error in generation queue", getattr(ex, "message", repr(ex)))
                    print("Error in generation queue", getattr(ex, "message", repr(ex)))
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

            # case for bertopic
            if e.kind == "bertopic":
                try:
                    bertmodel = cast(BertopicComputing, e)
                    print("Bertopic trained")
                    # self.bertopic.add(bertmodel)
                    logging.debug("Bertopic trained")
                except Exception as ex:
                    self.errors.append(
                        [datetime.now(config.timezone), "Error in bertopic training", str(ex)]
                    )
                    logging.error("Error in bertopic training", ex)
                finally:
                    self.computing.remove(e)
                    self.queue.delete(e.unique_id)

        # if there are predictions, add them
        for f in add_predictions:
            try:
                # load the prediction probabilities minus one
                df = pd.read_parquet(add_predictions[f])
                df = df.drop(columns=["entropy", "prediction", "dataset", "id", "label"])
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
                        datetime.now(config.timezone),
                        "Error in adding prediction",
                        str(ex),
                    ]
                )
                logging.error("Error in addind prediction", ex)

        # clean errors older than 15 minutes
        delta = datetime.now(config.timezone) - timedelta(minutes=15)
        self.errors = [error for error in self.errors if error[0] >= delta]

        return None

    def dump(self, with_files=True) -> None:
        """
        Dump the project in a archive
        - keep the files
        - do not keep the models

        Ideally, to be able to rerun everything
        """
        if self.params.dir is None:
            raise Exception("No directory for project")
        os.mkdir(self.params.dir.joinpath("dump"))

        # save the project parameters
        # - features computed

        # save the annotations

        # save the data (train + test + all)
        if with_files:
            shutil.copyfile(
                self.params.dir.joinpath("data_all.parquet"),
                self.params.dir.joinpath("dump").joinpath("data_all.parquet"),
            )
            shutil.copyfile(
                self.params.dir.joinpath("train.parquet"),
                self.params.dir.joinpath("dump").joinpath("data_train.parquet"),
            )
            if self.params.test:
                shutil.copyfile(
                    self.params.dir.joinpath("test.parquet"),
                    self.params.dir.joinpath("dump").joinpath("data_test.parquet"),
                )

        # save the codebook

        # create the archive
        shutil.make_archive(
            f"dump_{self.project_slug}", "zip", self.params.dir.joinpath("dump"), self.params.dir
        )

        # delete the dump folder
        shutil.rmtree(self.params.dir.joinpath("dump"))
        return None
