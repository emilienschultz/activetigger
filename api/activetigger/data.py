from pathlib import Path

import pandas as pd
from pandas import DataFrame


class Data:
    """
    Class to manage data

    TODO :
    - move update methods here
    - add methodes for data operations rather than using directly the DataFrame
    """

    path_project: Path
    path_data_all: Path
    path_datasets: Path
    train: DataFrame
    valid: DataFrame | None
    test: DataFrame | None
    formats_supported = [".csv", ".parquet", ".xlsx"]
    index: DataFrame | None = None

    def __init__(
        self,
        path_project: Path,
        path_data_all: Path,
        path_features: Path,
        path_train: Path,
        path_valid: Path | None,
        path_test: Path | None,
    ):
        self.path_project = path_project
        self.path_data_all = path_data_all
        self.path_features = path_features
        self.path_datasets = self.path_project.joinpath("data")

        if not self.path_datasets.exists():
            self.path_datasets.mkdir(parents=True, exist_ok=True)
        if not path_train.exists():
            raise FileNotFoundError(f"Training data file not found: {path_train}")
        if path_valid and not path_valid.exists():
            raise FileNotFoundError(f"Validation data file not found: {path_valid}")
        if path_test and not path_test.exists():
            raise FileNotFoundError(f"Test data file not found: {path_test}")

        self.train = pd.read_parquet(path_train)
        self.train["dataset"] = "train"
        if path_valid is not None:
            self.valid = pd.read_parquet(path_valid)
            self.valid["dataset"] = "valid"
        else:
            self.valid = None
        if path_test is not None:
            self.test = pd.read_parquet(path_test)
            self.test["dataset"] = "test"
        else:
            self.test = None
        self.index = None

    def check_format(self, filename: str) -> bool:
        """
        Check if the file format is supported
        """
        for ext in self.formats_supported:
            if filename.endswith(ext):
                return True
        return False

    def check_dataset_exists(self, dataset_name: str) -> bool:
        """
        Check if a dataset exists
        """
        return (self.path_datasets / dataset_name).exists()

    def read_dataset(self, filename: str) -> DataFrame:
        """
        Read a data file and return a DataFrame
        """
        file_path = self.path_datasets.joinpath(filename)
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if filename.endswith(".csv"):
            return pd.read_csv(file_path)
        elif filename.endswith(".parquet"):
            return pd.read_parquet(file_path)
        elif filename.endswith(".xlsx"):
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {filename}")

    def get_full_id(self) -> DataFrame:
        """
        Get the full list of IDs from the raw file
        """
        if self.index is None:
            self.index = pd.read_parquet(self.path_data_all, columns=["id_external"])
        return self.index

    def get_external_id(self, element_id: str) -> str:
        """
        Get the external ID for a given internal element ID
        """
        return str(self.get_full_id().loc[element_id, "id_external"])

    def get_datasets(self) -> DataFrame:
        """
        Get the list of available datasets
        """
        corpus = [self.train[["dataset"]]]
        if self.valid is not None:
            corpus.append(self.valid[["dataset"]])
        if self.test is not None:
            corpus.append(self.test[["dataset"]])
        df = pd.concat(corpus)
        return df
