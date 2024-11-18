import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series

from activetigger.db import DatabaseManager
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
    training: dict
    projections: dict
    possible_projections: dict
    options: dict
    lang: str
    db_manager: DatabaseManager
    n: int

    def __init__(
        self,
        project_slug: str,
        path_train: Path,
        path_all: Path,
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
        self.path_train = path_train
        self.path_all = path_all
        self.path_models = models_path
        self.queue = queue
        self.informations = {}
        self.map, self.n = self.get_map()
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
        self.db_manager.add_feature(
            project=self.project_slug,
            kind=kind,
            name=name,
            parameters=json.dumps(parameters),
            user=username,
            data=json.dumps(list(new_content.columns)),
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

        if self.db_manager.get_feature(self.project_slug, name) is None:
            return {"error": "feature doesn't exist in database"}

        col = self.get([name])
        # read data, delete columns and save
        content = pd.read_parquet(self.path_train)
        content[[i for i in content.columns if i not in col]].to_parquet(
            self.path_train
        )
        del content

        # delete from database
        self.db_manager.delete_feature(self.project_slug, name)

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
                    r = self.add(name, kind, username, parameters, df)
                    self.queue.delete(unique_id)
                    del self.training[name]
                    print("Add feature", name)

        # for projections
        training = [u for u in self.projections if "queue" in self.projections[u]]
        for u in training:
            unique_id = self.projections[u]["queue"]
            if self.queue.current[unique_id]["future"].done():
                try:
                    df = self.queue.current[unique_id]["future"].result()
                    self.projections[u]["data"] = df
                    self.projections[u]["id"] = self.projections[u]["queue"]
                    del self.projections[u]["queue"]
                    self.queue.delete(unique_id)
                except Exception as e:
                    print("Error in feature projections queue", e)
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
                return unique_id
            self.training[name] = {
                "unique_id": unique_id,
                "kind": kind,
                "parameters": parameters,
                "name": name,
                "username": username,
            }

        return {"success": "Feature in training"}
