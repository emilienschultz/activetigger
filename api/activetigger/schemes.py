import datetime
from io import StringIO
from typing import cast

import pandas as pd
from pandas import DataFrame
from sklearn.metrics import cohen_kappa_score  # type: ignore

from activetigger.datamodels import (
    AnnotationsDataModel,
    CodebookModel,
    CompareSchemesModel,
    SchemeModel,
    SchemesProjectStateModel,
    TableBatchInModel,
    TableOutModel,
)
from activetigger.db import DBException
from activetigger.db.manager import DatabaseManager
from activetigger.db.projects import ProjectsService
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
    db_manager: DatabaseManager
    content: DataFrame

    def __init__(
        self,
        project_slug: str,
        db_manager: DatabaseManager,
        train: DataFrame,
        valid: DataFrame | None = None,
        test: DataFrame | None = None,
    ) -> None:
        """
        Init empty
        """
        self.project_slug = project_slug
        self.projects_service = db_manager.projects_service
        self.db_manager = db_manager
        self.train = train
        self.valid = valid
        self.test = test

        available = self.available()

        # create a default scheme if not available
        if len(available) == 0:
            self.add_scheme(name="default", labels=[])

    def get_scheme_data(
        self,
        scheme: str,
        user: str | None = None,
        complete: bool = False,
        kind: list[str] = ["train"],
    ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        complete : add text from dataset & all the row
        """
        for k in kind:
            if k not in ["train", "test", "valid", "predict", "reconciliation"]:
                raise Exception(f"Kind {k} not recognized")
        if scheme not in self.available():
            raise Exception("Scheme doesn't exist")

        # get the current scheme
        results = self.projects_service.get_scheme_elements(self.project_slug, scheme, kind, user)
        df = pd.DataFrame(
            results, columns=["id", "dataset", "labels", "user", "timestamp", "comment"]
        ).set_index("id")

        df.index = pd.Index([str(i) for i in df.index])

        # only the labels
        if not complete:
            df["id"] = df.index
            return df

        # add the content from the datasets
        list_texts = []
        for k in kind:
            if k == "test":
                if self.test is not None:
                    list_texts.append(self.test[["text"]])
            elif k == "valid":
                if self.valid is not None:
                    list_texts.append(self.valid[["text"]])
            elif k == "train":
                if self.train is not None:
                    list_texts.append(self.train[["text"]])
        texts = pd.concat(list_texts)
        df = df.join(texts, rsuffix="_content", how="right")
        df["id"] = df.index
        return df

    def get_reconciliation_table(self, scheme: str) -> tuple[DataFrame, list[str]]:
        """
        Get reconciliation table
        TODO : manage different dataset
        TODO : it is pretty ugly, should be refactored
        """
        if scheme not in self.available():
            raise Exception("Scheme doesn't exist")

        results = self.projects_service.get_table_annotations_users(self.project_slug, scheme)
        # Shape the data
        df = pd.DataFrame(results, columns=["id", "labels", "user", "time"])  # shape as a dataframe

        # keep the real labels
        current_labels = df.loc[df.groupby("id")["time"].idxmax()].set_index("id")["labels"]

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
        df = df.join(self.train[["text"]], how="left")  # add the text

        df["current_label"] = current_labels
        df = df[f_multi].reset_index()

        # return the result
        return df, users

    def rename_label(self, former_label: str, new_label: str, scheme: str, username: str) -> None:
        """
        Convert tags from a specific label to another
        """
        # test if the new label exist, either create it
        if not self.exists_label(scheme, new_label):
            self.add_label(new_label, scheme, username)

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

        # add the new tags in train/test
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

    def get_total(self, dataset: str = "train") -> int:
        """
        Number of element in the dataset
        """
        if dataset == "test":
            # TODO: I think it should be tested way before now
            if self.test is None:
                raise Exception("Test dataset is not defined")
            return len(self.test)
        return len(self.train)

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
        batch: TableBatchInModel,
        user: str = "all",
    ) -> TableOutModel:
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
        if batch.mode not in ["tagged", "untagged", "all", "recent"]:
            batch.mode = "all"
        if batch.scheme not in self.available():
            raise Exception(f"Scheme {batch.scheme} is not available")

        # case of the test set, no fancy stuff
        df: DataFrame = self.get_scheme_data(
            batch.scheme,
            complete=True,
            kind=["test"] if batch.dataset == "test" else [batch.dataset],
        )
        # manage NaT to avoid problems with json
        df["timestamp"] = df["timestamp"].apply(lambda x: str(x) if pd.notna(x) else "")

        # case of recent annotations (no filter possible)
        if batch.mode == "recent":
            list_ids = self.projects_service.get_recent_annotations(
                self.project_slug, user, batch.scheme, batch.max - batch.min, batch.dataset
            )
            df_r = df.loc[list(list_ids)].fillna(" ")
            table = df_r.sort_index().fillna("")[
                ["id", "timestamp", "labels", "text", "comment", "user"]
            ]
            return TableOutModel(
                items=table.to_dict(orient="records"),
                total=len(table),
            )

        # build dataset
        if batch.mode == "tagged":
            df = df[df["labels"].notnull()]

        if batch.mode == "untagged":
            df = df[df["labels"].isnull()]

        # filter for contains
        if batch.contains:
            if batch.contains.startswith("ALL:") and len(batch.contains) > 4:
                contains_f = batch.contains.replace("ALL:", "")
                f_labels = df["labels"].str.contains(clean_regex(contains_f)).fillna(False)
                f_text = df["text"].str.contains(clean_regex(contains_f)).fillna(False)
                f_contains = f_labels | f_text
            else:
                f_contains = df["text"].str.contains(clean_regex(batch.contains))
            df = df[f_contains].fillna(False)

        # normalize size
        if batch.max == 0:
            batch.max = 20
        if batch.max > len(df):
            batch.max = len(df)

        if batch.min > len(df):
            raise Exception(
                f"Minimal value {batch.min} is too high. It should not exced the size of the data ({len(df)})"
            )

        table = (
            df.sort_index()
            .iloc[int(batch.min) : int(batch.max)]
            .fillna("")[["id", "timestamp", "labels", "text", "comment", "user"]]
        )

        print(df.head())

        return TableOutModel(
            items=table.to_dict(orient="records"),
            total=len(df),
        )

    def add_scheme(
        self,
        name: str,
        labels: list[str],
        kind: str = "multiclass",
        user: str = "server",
    ) -> None:
        """
        Add new scheme
        """
        if self.exists(name):
            raise Exception("Scheme already exists")
        self.projects_service.add_scheme(self.project_slug, name, labels, kind, user)

    def add_label(self, label: str, scheme: str, user: str) -> None:
        """
        Add label in a scheme
        """
        available = self.available()
        if (label is None) or (label == ""):
            raise Exception("Label cannot be empty")
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        if available[scheme] is None:
            raise Exception("Scheme is not defined")
        if label in available[scheme].labels:
            return None
        labels = available[scheme].labels
        labels.append(label)
        self.update_scheme(scheme, labels)

    def exists_label(self, scheme: str, label: str) -> bool:
        """
        Test if a label exist in a scheme
        """
        available = self.available()
        if scheme not in available:
            raise Exception("Scheme doesn't exist")
        if label in available[scheme].labels:
            return True
        return False

    def delete_label(self, label: str, scheme: str, user: str) -> None:
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

    def update_scheme(self, scheme: str, labels: list) -> None:
        """
        Update existing schemes from database
        """
        self.projects_service.update_scheme_labels(self.project_slug, scheme, labels)

    def duplicate_scheme(self, scheme_name: str, new_scheme_name: str, username: str) -> None:
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

    def rename_scheme(self, old_name: str, new_name: str) -> None:
        """
        Rename a scheme
        """
        schemes = self.available()

        if old_name not in schemes:
            raise Exception("Scheme does not exist")
        if new_name in schemes:
            raise Exception("New name already exists")

        self.projects_service.rename_scheme(self.project_slug, old_name, new_name)

    def delete_scheme(self, name) -> None:
        """
        Delete a scheme
        """
        schemes = self.available()
        if name not in schemes:
            raise Exception("Scheme does not exist")
        if len(schemes) == 1:
            raise Exception("Cannot delete the last scheme")

        self.projects_service.delete_scheme(self.project_slug, name)

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
        return {"project_slug": self.project_slug, "availables": self.available()}

    def delete_annotation(
        self, element_id: str, scheme: str, dataset: str, user: str = "server"
    ) -> None:
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

        if a[scheme].kind == "multiclass":
            if label not in a[scheme].labels and label is not None:
                raise Exception(f"Label {label} not in the scheme")

        elif a[scheme].kind == "multilabel" and label is not None:
            er = [i for i in label.split("|") if i not in a[scheme].labels]
            if len(er) > 0:
                raise Exception(f"Labels {er} not in the scheme")
        elif a[scheme].kind == "span":
            print("Span annotation, no label check for the moment")

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
        return self.db_manager.users_service.get_coding_users(scheme, self.project_slug)

    def add_codebook(self, scheme: str, codebook: str, time: str) -> None:
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

    def get_codebook(self, scheme: str) -> CodebookModel:
        """
        Get codebook
        """
        try:
            r = self.projects_service.get_scheme_codebook(self.project_slug, scheme)
            return CodebookModel(
                scheme=scheme,
                content=str(r["codebook"]),
                time=str(r["time"]),
            )
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

    def add_file_annotations(
        self, annotationsdata: AnnotationsDataModel, user: str, dataset: str
    ) -> None:
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
        common_id = [i for i in col.index if i in self.train.index]

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

    def state(self) -> SchemesProjectStateModel:
        """
        Get the state of the project
        """
        return SchemesProjectStateModel(available=self.available())

    def compare(self, schemeA: str, schemeB: str) -> CompareSchemesModel:
        """
        Compare two schemes
        """

        labels = self.available()
        if schemeA not in labels:
            raise Exception("Scheme A doesn't exist")
        if schemeB not in labels:
            raise Exception("Scheme B doesn't exist")

        schemeA_labels = labels[schemeA].labels
        schemeB_labels = labels[schemeB].labels

        # proportion of similar labels
        labels_overlapping = round(
            100
            * len([i for i in schemeA_labels if i in schemeB_labels])
            / len(set(schemeA_labels + schemeB_labels)),
            2,
        )

        df_A = self.get_scheme_data(schemeA)
        df_B = self.get_scheme_data(schemeB)

        # only keeps elements that have been annotated in both schemes
        df = pd.concat({"schemeA": df_A["labels"], "schemeB": df_B["labels"]}, axis=1).dropna()

        # compute scores
        n_overlapping_annotations = len(df)
        score_ck = None
        percentage = 0
        if n_overlapping_annotations > 0:
            score_ck = cohen_kappa_score(df["schemeA"], df["schemeB"])
            percentage = len(df[df["schemeA"] == df["schemeB"]]) / n_overlapping_annotations  # type: ignore

        return CompareSchemesModel(
            datetime=datetime.datetime.now(),
            project_slug=self.project_slug,
            schemeA=schemeA,
            schemeB=schemeB,
            labels_overlapping=labels_overlapping,
            n_annotated=n_overlapping_annotations,
            cohen_kappa=score_ck,
            percentage=percentage,
        )
