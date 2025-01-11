import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series

from activetigger.db.projects import ProjectsService
from activetigger.functions import to_dtm, to_fasttext, to_sbert
from activetigger.queue import Queue

# Use parquet files to save features
# In the future : database ?


class Features:
    """
    Manage project features
    Comment :
    - for the moment as a file
    - database for informations
    - use "__" as separator
    """

    project_slug: str
    path_train: Path
    path_model: Path
    path_all: Path
    queue: Queue
    informations: dict
    content: DataFrame
    map: dict
    computing: dict
    options: dict
    lang: str
    projects_service: ProjectsService
    n: int

    def __init__(
        self,
        project_slug: str,
        path_train: Path,
        path_all: Path,
        models_path: Path,
        queue: Any,
        computing: dict,
        db_manager,
        lang: str,
    ) -> None:
        """
        Initit features
        """
        self.project_slug = project_slug
        self.projects_service = db_manager.projects_service
        self.path_train = path_train
        self.path_all = path_all
        self.path_models = models_path
        self.queue = queue
        self.informations = {}
        self.map, self.n = self.get_map()
        self.lang = lang
        self.computing = computing

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
            "dataset": {},
        }

        self.get_map()

    def __repr__(self) -> str:
        return f"Available features : {self.map}"

    def get_map(self) -> tuple[dict, int]:
        parquet_file = pq.ParquetFile(self.path_train)
        column_names = parquet_file.schema.names

        def find_strings_with_pattern(strings, pattern):
            matching_strings = [s for s in strings if re.match(pattern, s)]
            return matching_strings

        var = set(
            [i.split("__")[0] for i in column_names if "__index" not in i and i != "id"]
        )
        dic = {i: find_strings_with_pattern(column_names, i) for i in var}
        num_rows = parquet_file.metadata.num_rows
        return dic, num_rows

    def add(
        self,
        name: str,
        kind: str,
        username: str,
        parameters: dict,
        new_content: DataFrame | Series,
    ) -> dict:
        """
        Add feature(s) and save
        """
        # test name
        if name in self.map:
            return {"error": "feature name already exists for this project"}

        # test length
        if len(new_content) != self.n:
            raise ValueError("Features don't have the right shape")

        # change type for series
        if type(new_content) is Series:
            new_content = pd.DataFrame(new_content)

        # change column name with a prefix
        new_content.columns = [f"{name}__{i}" for i in new_content.columns]

        # read data, add the feature to the dataset and save
        content = pd.read_parquet(self.path_train)
        content = pd.concat(
            [
                content[[i for i in content.columns if i not in new_content.columns]],
                new_content,
            ],
            axis=1,
        )
        content.to_parquet(self.path_train)
        del content

        # add informations to database
        self.projects_service.add_feature(
            project=self.project_slug,
            kind=kind,
            name=name,
            parameters=parameters,
            user=username,
            data=list(new_content.columns),
        )

        # refresh the map
        self.map = self.get_map()[0]

        return {"success": "feature added"}

    def delete(self, name: str):
        """
        Delete feature
        """
        if name not in self.map:
            return {"error": "feature doesn't exist in mapping"}

        if self.projects_service.get_feature(self.project_slug, name) is None:
            return {"error": "feature doesn't exist in database"}

        col = self.get([name])
        # read data, delete columns and save
        content = pd.read_parquet(self.path_train)
        content[[i for i in content.columns if i not in col]].to_parquet(
            self.path_train
        )
        del content

        # delete from database
        self.projects_service.delete_feature(self.project_slug, name)

        # refresh the map
        self.map = self.get_map()[0]

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

        # load only needed data from file
        data = pd.read_parquet(self.path_train, columns=cols)

        return data

    def info(self, name: str):
        feature = self.projects_service.get_feature(self.project_slug, name)
        if feature is None:
            return {"error": "feature doesn't exist in database"}
        return {
            "time": feature.time,
            "name": name,
            "kind": feature.kind,
            "username": feature.user,
            "parameters": feature.parameters,
            "columns": feature.data,
        }

    def get_available(self):
        """
        Informations on features + update
        Comments:
            Maybe not the best solution
            Database ? How to avoid a loop ...
        """
        features = self.projects_service.get_project_features(self.project_slug)
        return features

    def get_column_raw(self, column_name: str, index: str = "train") -> dict:
        """
        Get column raw dataset
        """
        df = pd.read_parquet(self.path_all)
        df_train = pd.read_parquet(self.path_train, columns=[])  # only the index
        if column_name not in list(df.columns):
            return {"error": "Column doesn't exist"}
        if index == "train":  # filter only train id
            return {"success": df.loc[df_train.index][column_name]}
        elif index == "all":
            return {"success": df[column_name]}
        else:
            return {"error": "Wrong index"}

    def current_user_processes(self, user: str):
        return [e for e in self.computing if e["user"] == user]

    def current_computing(self):
        return [e["name"] for e in self.computing if e["kind"] == "feature"]

    def compute(
        self, df: pd.Series, name: str, kind: str, parameters: dict, username: str
    ):
        """
        Compute new feature
        """
        if len(self.current_user_processes(username)) > 0:
            return {"error": "There is already a process launched by the user"}

        if kind not in {"sbert", "fasttext", "dfm", "regex", "dataset"}:
            return {"error": "Not implemented"}

        # different types of features
        if kind == "regex":
            if "value" not in parameters:
                return {"error": "Parameters missing for the regex"}

            regex_name = f"regex_[{parameters['value']}]_by_{username}"
            pattern = re.compile(parameters["value"])
            f = df.apply(lambda x: bool(pattern.search(x)))
            parameters["count"] = int(f.sum())
            r = self.add(regex_name, kind, username, parameters, f)
            return {"success": "regex added"}

        elif kind == "dataset":
            # get the raw column for the train set
            r = self.get_column_raw(parameters["dataset_col"])
            if "error" in r:
                return r
            column = r["success"]

            # convert the column to a specific format
            if len(column.dropna()) != len(column):
                return {"error": "Column contains null values"}
            if parameters["dataset_type"] == "Numeric":
                try:
                    column = column.apply(float)
                except Exception:
                    return {
                        "error": "The column can't be transform into numerical feature"
                    }
            else:
                column = column.apply(str)

            # add the feature to the project
            dataset_name = f"dataset_{parameters['dataset_col']}_{parameters['dataset_type']}".lower()
            self.add(dataset_name, kind, username, parameters, column)
            return {"success": "Feature added"}
        else:
            if kind == "sbert":
                args = {"texts": df, "model": "distiluse-base-multilingual-cased-v1"}
                func = to_sbert
            elif kind == "fasttext":
                args = {
                    "texts": df,
                    "language": self.lang,
                    "path_models": self.path_models,
                }
                func = to_fasttext

            elif kind == "dfm":
                args = parameters.copy()
                args["texts"] = df
                args["language"] = self.lang
                func = to_dtm

            # add the computation to queue
            unique_id = self.queue.add("feature", func, args)
            if unique_id == "error":
                return "error"

            self.computing.append(
                {
                    "unique_id": unique_id,
                    "kind": "feature",
                    "parameters": parameters,
                    "type": kind,
                    "user": username,
                    "name": name,
                    "time": datetime.now(),
                }
            )

        return {"success": "Feature in training"}
