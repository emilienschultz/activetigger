from io import StringIO
from pathlib import Path
from typing import Any, cast

import pandas as pd
from pandas import DataFrame

from activetigger.datamodels import (
    AnnotationsDataModel,
    SchemeModel,
    SchemesProjectStateModel,
    TableBatch,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.db.projects import Codebook, ProjectsService
from activetigger.functions import clean_regex, slugify


class Schemes:
    """
    Manage project schemes & tags

    Tables :
    - schemes
    - annotations
    """

    project_slug: str
    projects_service: ProjectsService
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
        self.projects_service = db_manager.projects_service
        self.db_manager = db_manager
        self.content = pd.read_parquet(path_content)  # text + context
        if path_test.exists():
            self.test = pd.read_parquet(path_test)
        else:
            self.test = None

        available = self.available()

        # create a default scheme if not available
        if len(available) == 0:
            self.add_scheme(name="default", labels=[])

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available()}"

    def get_scheme_data(
        self, scheme: str, complete: bool = False, kind: list[str] = ["train"]
    ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        Join with text data in separate file (train or test, in this case it is a XOR)

        Comments:
            For the moment tags can be add, test, predict, reconciliation
        """
        if kind is None:
            kind = ["train"]
        if scheme not in self.available():
            raise Exception("Scheme doesn't exist")

        if isinstance(kind, str):
            kind = [kind]

        # get all elements from the db
        # - last element for each id
        # - for a specific scheme

        results = self.projects_service.get_scheme_elements(self.project_slug, scheme, kind)

        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "timestamp", "comment"]
        ).set_index("id")
        df.index = [str(i) for i in df.index]
        if complete:  # all the elements
            if "test" in kind:
                if len(kind) > 1:
                    raise Exception("Cannot ask for both train and test")
                # case if the test, join the text data
                t = self.test[["text"]].join(df)
                return t
            else:
                return self.content.join(df, rsuffix="_content")
        return df

    def get_reconciliation_table(self, scheme: str):
        """
        Get reconciliation table
        TODO : add the filter on action
        """
        if scheme not in self.available():
            raise Exception("Scheme doesn't exist")

        results = self.projects_service.get_table_annotations_users(self.project_slug, scheme)
        # Shape the data
        df = pd.DataFrame(results, columns=["id", "labels", "user", "time"])  # shape as a dataframe

        def agg(x):
            return list(x)[0] if len(x) > 0 else None  # take the label else None

        df = df.pivot_table(
            index="id", columns="user", values="labels", aggfunc=agg
        )  # pivot and keep the label
        f_multi = df.apply(
            lambda x: len(set([i for i in x if pd.notna(i)])) > 1, axis=1
        )  # filter for disagreement
        users = list(df.columns)
        df = pd.DataFrame(df.apply(lambda x: x.to_dict(), axis=1), columns=["annotations"])
        df = df.join(self.content[["text"]], how="left")  # add the text
        df = df[f_multi].reset_index()
        # return the result
        return df, users

    def convert_annotations(
        self, former_label: str, new_label: str, scheme: str, username: str
    ) -> None:
        """
        Convert tags from a specific label to another
        """
        # add a new tag for the annotated id in the trainset
        df_train = self.get_scheme_data(scheme, kind=["train"])
        elements_train = [
            {"element_id": element_id, "annotation": new_label, "comment": "label renamed"}
            for element_id in list(df_train[df_train["labels"] == former_label].index)
        ]
        df_test = self.get_scheme_data(scheme, kind=["test"])
        elements_test = [
            {"element_id": element_id, "annotation": new_label, "comment": "label renamed"}
            for element_id in list(df_test[df_test["labels"] == former_label].index)
        ]

        # add the new tags
        self.db_manager.projects_service.add_annotations(
            dataset="train",
            user_name=username,
            project_slug=self.project_slug,
            scheme=scheme,
            elements=elements_train,
        )
        self.db_manager.projects_service.add_annotations(
            dataset="test",
            user_name=username,
            project_slug=self.project_slug,
            scheme=scheme,
            elements=elements_test,
        )

        # update the scheme (no need to add empty annotation in the database)
        available = self.available()
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        labels = available[scheme].labels
        labels.remove(former_label)
        self.update_scheme(scheme, labels)

    def get_total(self, dataset: str = "train"):
        """
        Number of element in the dataset
        """
        if dataset == "test":
            # TODO: I think it should be tested way before now
            if self.test is None:
                raise Exception("Test dataset is not defined")
            return len(self.test)
        return len(self.content)

    def get_sample(
        self,
        scheme: str,
        n_elements: int,
        mode: str,
        dataset: str = "train",
        random: bool = False,
    ) -> DataFrame:
        """
        Get a sample of element following a method
        """
        if mode not in ["tagged", "untagged", "all"]:
            raise Exception("Mode not available")
        if scheme not in self.available():
            raise Exception("Scheme doesn't exist")
        df = self.get_scheme_data(scheme, complete=True, kind=[dataset])
        # build dataset
        if mode == "tagged":
            df = cast(DataFrame, df[df["labels"].notnull()])

        if mode == "untagged":
            df = cast(DataFrame, df[df["labels"].isnull()])

        if n_elements > len(df):
            n_elements = len(df)

        if random:
            return df.sample(n=n_elements)
        return df.head(n_elements)

    def get_table(
        self,
        scheme: str,
        min: int,
        max: int,
        mode: str,
        contains: str | None = None,
        dataset: str = "train",
        user: str = "all",
    ) -> TableBatch:
        """
        Get data table
        scheme : the annotations
        min, max: the range
        mode: sample
        contains: search

        set: train or test

        Choice to order by index.
        """
        # check for errors
        if mode not in ["tagged", "untagged", "all", "recent"]:
            mode = "all"
        if scheme not in self.available():
            raise Exception(f"Scheme {scheme} is not available")

        # case of the test set, no fancy stuff
        df: DataFrame = self.get_scheme_data(
            scheme, complete=True, kind=["test"] if dataset == "test" else [dataset]
        )

        # case of recent annotations (no filter possible)
        if mode == "recent":
            list_ids = self.projects_service.get_recent_annotations(
                self.project_slug, user, scheme, max - min, dataset
            )
            df_r = cast(DataFrame, df.loc[list(list_ids)].reset_index())
            return TableBatch(
                batch=df_r,
                total=len(df_r),
                min=0,
                max=len(df_r),
                filter="recent",
            )

        # build dataset
        if mode == "tagged":
            df = cast(DataFrame, df[df["labels"].notnull()])

        if mode == "untagged":
            df = cast(DataFrame, df[df["labels"].isnull()])

        # filter for contains
        if contains:
            print(contains)
            if contains.startswith("ALL:") and len(contains) > 4:
                contains_f = contains.replace("ALL:", "")
                f_labels = df["labels"].str.contains(clean_regex(contains_f)).fillna(False)
                f_text = df["text"].str.contains(clean_regex(contains_f)).fillna(False)
                f_contains = f_labels | f_text
            else:
                f_contains = df["text"].str.contains(clean_regex(contains))
            df = cast(DataFrame, df[f_contains]).fillna(False)

        # normalize size
        if max == 0:
            max = 20
        if max > len(df):
            max = len(df)

        if min > len(df):
            raise Exception(
                f"Minimal value {min} is too high. It should not exced the size of the data ({len(df)})"
            )

        return TableBatch(
            batch=df.sort_index().iloc[min:max].reset_index(),
            total=len(df),
            min=min,
            max=max,
            filter=contains,
        )

    def add_scheme(
        self,
        name: str,
        labels: list[str],
        kind: str = "multiclass",
        user: str = "server",
    ):
        """
        Add new scheme
        """
        if self.exists(name):
            raise Exception("Scheme already exists")

        self.projects_service.add_scheme(self.project_slug, name, labels, kind, user)

        return {"success": "scheme created"}

    def add_label(self, label: str, scheme: str, user: str):
        """
        Add label in a scheme
        """
        available = self.available()
        print(available)
        if (label is None) or (label == ""):
            raise Exception("Label cannot be empty")
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        if available[scheme] is None:
            available[scheme] = []
        if label in available[scheme].labels:
            return {"success": "label already in the scheme"}
        labels = available[scheme].labels
        labels.append(label)
        self.update_scheme(scheme, labels)
        return {"success": "scheme updated with a new label"}

    def exists_label(self, scheme: str, label: str):
        """
        Test if a label exist in a scheme
        """
        available = self.available()
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        if label in available[scheme].labels:
            return True
        return False

    def delete_label(self, label: str, scheme: str, user: str):
        """
        Delete a label in a scheme
        """
        available = self.available()
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        if label not in available[scheme].labels:
            raise Exception("Label doesn't exist")
        labels = available[scheme].labels
        labels.remove(label)
        # push empty entry for tagged elements
        # both for train
        df = self.get_scheme_data(scheme, kind=["train"])
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            self.push_annotation(i, None, scheme, user, "train", "delete")
        # and test
        df = self.get_scheme_data(scheme, kind=["test"])
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            self.push_annotation(i, None, scheme, user, "test", "delete")
        # update scheme
        self.update_scheme(scheme, labels)
        return {"success": "scheme updated removing a label"}

    def update_scheme(self, scheme: str, labels: list):
        """
        Update existing schemes from database
        """
        self.projects_service.update_scheme_labels(self.project_slug, scheme, labels)
        return {"success": "scheme updated"}

    def duplicate_scheme(self, scheme_name: str, new_scheme_name: str, username: str):
        """
        Duplicate a scheme
        """

        schemes = self.available()

        if scheme_name not in schemes:
            raise Exception("Scheme does not exist")
        if new_scheme_name in schemes:
            raise Exception("New name already exists")

        self.projects_service.duplicate_scheme(
            self.project_slug, scheme_name, new_scheme_name, username
        )

    def rename_scheme(self, old_name: str, new_name: str):
        """
        Rename a scheme
        """

        schemes = self.available()

        if old_name not in schemes:
            raise Exception("Scheme does not exist")
        if new_name in schemes:
            raise Exception("New name already exists")

        self.projects_service.rename_scheme(self.project_slug, old_name, new_name)

    def delete_scheme(self, name) -> dict:
        """
        Delete a scheme
        """

        schemes = self.available()

        if name not in schemes:
            raise Exception("Scheme does not exist")
        if len(schemes) == 1:
            raise Exception("Cannot delete the last scheme")

        self.projects_service.delete_scheme(self.project_slug, name)
        return {"success": "scheme deleted"}

    def exists(self, name: str) -> bool:
        """
        Test if scheme exist
        """
        if name in self.available():
            return True
        return False

    def available(self) -> dict[str, SchemeModel]:
        """
        Available schemes {scheme:[labels]}
        """
        r = self.projects_service.available_schemes(self.project_slug)
        return {
            i["name"]: SchemeModel(
                name=i["name"], labels=i["labels"], kind=i["kind"], project_slug=self.project_slug
            )
            for i in r
        }

    def get(self) -> dict:
        """
        state of the schemes
        """
        r = {"project_slug": self.project_slug, "availables": self.available()}
        return r

    def delete_annotation(
        self, element_id: str, scheme: str, dataset: str | None, user: str = "server"
    ) -> bool:
        """
        Delete a recorded tag
        i.e. : add empty label
        """

        self.projects_service.add_annotation(
            dataset="delete",
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=None,
        )
        self.projects_service.add_annotation(
            dataset=dataset,
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=None,
        )

        return True

    def push_annotation(
        self,
        element_id: str,
        label: str | None,
        scheme: str,
        user: str = "server",
        mode: str | None = "train",
        comment: str | None = "",
        selection: str | None = None,
    ) -> None:
        """
        Record a tag in the database
        mode : train, predict, test
        """

        if element_id == "noelement":
            raise Exception("No element id")

        if mode is None:
            mode = "undefined"

        # test if the action is possible
        a = self.available()
        if scheme not in a:
            raise Exception("Scheme doesn't exist")

        # test if the labels used exist in the scheme
        if label is None:
            print("Add null label for ", element_id)
        elif "|" in label:
            er = [i for i in label.split("|") if i not in a[scheme].labels]
            if len(er) > 0:
                raise Exception(f"Labels {er} not in the scheme")
        else:
            if label not in a[scheme].labels:
                raise Exception(f"Label {label} not in the scheme")

        self.projects_service.add_annotation(
            dataset=mode,
            user_name=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=label,
            comment=comment,
            selection=selection,
        )

    def get_coding_users(self, scheme: str) -> list[str]:
        """
        Get users action for a scheme
        """
        results = self.db_manager.users_service.get_coding_users(scheme, self.project_slug)
        return results

    def add_codebook(self, scheme: str, codebook: str, time: str):
        """
        Add codebook
        if mismatch between date, keep both and return error
        """
        # get lastmodified timestamp for the scheme
        r = self.projects_service.get_scheme_codebook(self.project_slug, scheme)
        # if no modification since the last time, ok
        if r["time"] == time:
            try:
                self.projects_service.update_scheme_codebook(self.project_slug, scheme, codebook)
            except DBException as e:
                raise Exception("Codebook not added") from e
            return {"success": "Codebook added"}
        # if scheme have been modified since the last time
        else:
            new_codebook = f"""
# [CONFLICT] -------- NEW CODEBOOK --------

{codebook}

# [CONFLICT] -------- PREVIOUS CODEBOOK --------

{r["codebook"]}"""
            try:
                self.projects_service.update_scheme_codebook(
                    self.project_slug, scheme, new_codebook
                )
            except DBException as e:
                raise Exception("Codebook not added") from e
            raise Exception("Codebook in conflict, please refresh and arbitrate")

    def get_codebook(self, scheme: str) -> Codebook:
        """
        Get codebook
        """
        try:
            return self.projects_service.get_scheme_codebook(self.project_slug, scheme)
        except DBException as e:
            raise Exception from e

    def dichotomize(self, annotation: str | None, label: str | None) -> str:
        """
        check if the label is in the annotation
        current situation : separator |
        """
        if annotation is None:
            raise Exception("No annotation")
        if label is None:
            raise Exception("No label")
        return label if label in annotation.split("|") else "not-" + label

    def add_file_annotations(self, annotationsdata: AnnotationsDataModel, user: str, dataset: str):
        """
        Add annotations from a file
        Create labels if not exist
        """
        # check if the scheme exist
        if annotationsdata.scheme not in self.available():
            raise Exception("Scheme doesn't exist")
        else:
            labels = self.available()[annotationsdata.scheme].labels

        # convert the data, slugiy the index, set the index, drop empty
        df = pd.read_csv(StringIO(annotationsdata.csv))
        df[annotationsdata.col_id] = df[annotationsdata.col_id].astype(str).apply(slugify)
        if len(set(df[annotationsdata.col_id])) != len(df):
            raise Exception("Duplicate IDs after slugify in the column selected as ID")
        df = df.set_index(annotationsdata.col_id)
        df = df[df[annotationsdata.col_label].notna()]
        col = df[annotationsdata.col_label]

        # only elements existing in the dataset
        common_id = [i for i in col.index if i in self.content.index]

        # if needed, create the labels in the scheme
        for i in col.unique():
            if i not in labels:
                self.add_label(i, annotationsdata.scheme, user)

        # create the elements to add
        elements_test = [
            {"element_id": element_id, "annotation": label, "comment": ""}
            for element_id, label in col.loc[common_id].items()
        ]

        # add the new tags in batch
        self.db_manager.projects_service.add_annotations(
            dataset="train",
            user_name=user,
            project_slug=self.project_slug,
            scheme=annotationsdata.scheme,
            elements=elements_test,
            selection="import",
        )

        if len(common_id) < len(df):
            print(
                f"Some elements annoted in the dataset where not added (index mismatch) or not in the trainset. \
                    Number of elements added : {len(common_id)} (total annotated : {len(df)})"
            )

        # TODO : return message ?

    def state(self) -> SchemesProjectStateModel:
        """
        Get the state of the project
        """
        return SchemesProjectStateModel(available=self.available())
