import json
import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame, Series

from activetigger.db import DatabaseManager
from activetigger.functions import to_dtm, to_fasttext, to_sbert
from activetigger.queue import Queue


class Features:
    """
    Manage project features
    Comment :
    - for the moment as a file
    - database for informations
    - use "__" as separator
    """

    project_slug: str
    path: Path
    path_model: Path
    path_raw: Path
    queue: Queue
    informations: dict
    content: DataFrame
    map: dict
    training: dict
    projections: dict
    possible_projections: dict
    options: dict
    lang: str
    db_manager: DatabaseManager

    def __init__(
        self,
        project_slug: str,
        data_path: Path,
        raw_path: Path,
        models_path: Path,
        queue,
        db_manager,
        lang: str,
    ) -> None:
        """
        Initit features
        """
        self.project_slug = project_slug
        self.db_manager = db_manager
        self.path = data_path
        self.path_raw = raw_path
        self.path_models = models_path
        self.queue = queue
        self.informations = {}
        content, map = self.load()
        self.content = content
        self.map = map
        self.training: dict = {}
        self.lang = lang

        # managing projections
        self.projections: dict = {}
        self.possible_projections: dict = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": ["cosine", "euclidean"],
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
            "dataset": {},
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

    def add(
        self,
        name: str,
        kind: str,
        username: str,
        parameters: dict,
        content: DataFrame | Series,
    ) -> dict:
        """
        Add feature(s) and save
        """
        # test name
        if name in self.map:
            return {"error": "feature name already exists for this project"}

        # test length
        if len(content) != len(self.content):
            raise ValueError("Features don't have the right shape")

        # change type for series
        if type(content) is Series:
            content = pd.DataFrame(content)

        # change column name with a prefix
        content.columns = [f"{name}__{i}" for i in content.columns]

        # add the mapping
        self.map[name] = list(content.columns)

        # add the feature to the dataset and save
        self.content = pd.concat([self.content, content], axis=1)
        self.content.to_parquet(self.path)

        # add informations to database
        self.db_manager.add_feature(
            self.project_slug,
            kind,
            name,
            json.dumps(parameters),
            username,
            json.dumps(list(content.columns)),
        )
        return {"success": "feature added"}

    def delete(self, name: str):
        """
        Delete feature
        """
        if name not in self.map:
            return {"error": "feature doesn't exist in mapping"}

        if self.db_manager.get_feature(self.project_slug, name) is None:
            return {"error": "feature doesn't exist in database"}

        col = self.get([name])
        del self.map[name]
        self.content = self.content.drop(columns=col)
        self.content.to_parquet(self.path)
        self.db_manager.delete_feature(self.project_slug, name)
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

    def info(self, name: str):
        feature = self.db_manager.get_feature(self.project_slug, name)
        if feature is None:
            return {"error": "feature doesn't exist in database"}
        return {
            "time": feature.time,
            "name": name,
            "kind": feature.kind,
            "username": feature.user,
            "parameters": json.loads(feature.parameters),
            "columns": json.loads(feature.data),
        }

    def update_processes(self):
        """
        Check for computing processing completed
        and clean them for the queue
        """
        # for features
        for name in self.training.copy():
            unique_id = self.training[name]["unique_id"]

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
                    kind = self.training[name]["kind"]
                    parameters = self.training[name]["parameters"]
                    username = self.training[name]["username"]
                    self.add(name, kind, username, parameters, df)
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

    def get_available(self):
        """
        Informations on features + update
        Comments:
            Maybe not the best solution
            Database ? How to avoid a loop ...
        """
        features = self.db_manager.get_project_features(self.project_slug)
        return features

    def get_column_raw(self, column_name: str) -> dict:
        """
        Get column raw dataset
        """
        df = pd.read_parquet(self.path_raw)
        if column_name not in list(df.columns):
            return {"error": "Column doesn't exist"}
        # filter only train id
        return {"success": df.loc[self.content.index][column_name]}

    def compute(
        self, df: pd.Series, name: str, kind: str, parameters: dict, username: str
    ):
        """
        Compute new feature
        """
        if name in self.training:
            return {"error": "feature already in training"}

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
            self.add(regex_name, kind, username, parameters, f)
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
                args = parameters
                args["texts"] = df
                func = to_dtm

            # add the computation to queue
            unique_id = self.queue.add("feature", func, args)
            if unique_id == "error":
                return unique_id
            self.training[name] = {
                "unique_id": unique_id,
                "kind": kind,
                "parameters": parameters,
                "name": name,
                "username": username,
            }

        return {"success": "Feature in training"}
