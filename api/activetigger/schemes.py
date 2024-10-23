import json
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from activetigger.datamodels import TableBatch
from activetigger.db import DatabaseManager


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
        self, scheme: str, complete: bool = False, kind: list | str = ["train"]
    ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        Join with text data in separate file (train or test, in this case it is a XOR)

        Comments:
            For the moment tags can be add, test, predict, reconciliation
        """
        if scheme not in self.available():
            return {"error": "Scheme doesn't exist"}

        if isinstance(kind, str):
            kind = [kind]

        # get all elements from the db
        # - last element for each id
        # - for a specific scheme

        results = self.db_manager.get_scheme_elements(self.project_slug, scheme, kind)

        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "timestamp"]
        ).set_index("id")
        df.index = [str(i) for i in df.index]
        if complete:  # all the elements
            if "test" in kind:
                if len(kind) > 1:
                    return {"error": "Test data cannot be mixed with train data"}
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
        if scheme not in self.available():
            return {"error": "Scheme doesn't exist"}

        results = self.db_manager.get_table_annotations_users(self.project_slug, scheme)
        # Shape the data
        df = pd.DataFrame(
            results, columns=["id", "labels", "user", "time"]
        )  # shape as a dataframe

        def agg(x):
            return list(x)[0] if len(x) > 0 else None  # take the label else None

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

    def convert_annotations(
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
            self.push_annotation(i, new_label, scheme, username, "train")
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
    ) -> TableBatch:
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
        if mode not in ["tagged", "untagged", "all", "recent"]:
            mode = "all"
        if scheme not in self.available():
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

        return {
            "batch": df.sort_index().iloc[min:max].reset_index(),
            "total": len(df),
            "min": min,
            "max": max,
            "filter": contains,
        }

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
        if scheme not in available:
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
        if scheme not in available:
            return {"error": "scheme doesn't exist"}
        if label in available[scheme]:
            return True
        return False

    def delete_label(self, label: str, scheme: str, user: str):
        """
        Delete a label in a scheme
        """
        available = self.available()
        if scheme not in available:
            return {"error": "scheme doesn't exist"}
        if label not in available[scheme]:
            return {"error": "label does not exist"}
        labels = available[scheme]
        labels.remove(label)
        # push empty entry for tagged elements
        # both for train
        df = self.get_scheme_data(scheme, kind="train")
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            print(i)
            self.push_annotation(i, None, scheme, user, "train")
        # and test
        df = self.get_scheme_data(scheme, kind="test")
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            print(i)
            self.push_annotation(i, None, scheme, user, "test")
        # update scheme
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

    def delete_annotation(
        self, element_id: str, scheme: str, dataset: str, user: str = "server"
    ) -> bool:
        """
        Delete a recorded tag
        i.e. : add empty label
        """

        self.db_manager.add_annotation(
            action="delete",
            user=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=None,
        )
        self.db_manager.add_annotation(
            action=dataset,
            user=user,
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
        mode: str = "train",
    ):
        """
        Record a tag in the database
        mode : train, predict, test
        """

        # test if the action is possible
        a = self.available()
        if scheme not in a:
            return {"error": "scheme unavailable"}
        if (label is not None) and (label not in a[scheme]):
            return {"error": "this tag doesn't belong to this scheme"}

        # TODO : add a test also for testing
        # if (not element_id in self.content.index):
        #    return {"error":"element doesn't exist"}

        print("PUSH TAG", mode, user, self.project_slug, element_id, scheme, label)
        self.db_manager.add_annotation(
            action=mode,
            user=user,
            project_slug=self.project_slug,
            element_id=element_id,
            scheme=scheme,
            annotation=label,
        )
        print(
            (
                "push annotation",
                mode,
                user,
                self.project_slug,
                element_id,
                scheme,
                label,
            )
        )
        return {"success": "annotation added"}

    def get_coding_users(self, scheme: str):
        """
        Get users action for a scheme
        """
        results = self.db_manager.get_coding_users(scheme, self.project_slug)
        return results
