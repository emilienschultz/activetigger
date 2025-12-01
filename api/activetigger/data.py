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

    project_dir: Path
    datasets_dir: Path
    train: DataFrame
    valid: DataFrame | None
    test: DataFrame | None
    formats_supported = [".csv", ".parquet", ".xlsx"]

    def __init__(
        self, project_dir: Path, path_train: Path, path_valid: Path | None, path_test: Path | None
    ):
        self.project_dir = project_dir
        self.datasets_dir = self.project_dir.joinpath("data")
        if not self.datasets_dir.exists():
            self.datasets_dir.mkdir(parents=True, exist_ok=True)
        if not path_train.exists():
            raise FileNotFoundError(f"Training data file not found: {path_train}")
        if path_valid and not path_valid.exists():
            raise FileNotFoundError(f"Validation data file not found: {path_valid}")
        if path_test and not path_test.exists():
            raise FileNotFoundError(f"Test data file not found: {path_test}")

        self.train = pd.read_parquet(path_train)
        self.valid = pd.read_parquet(path_valid) if path_valid else None
        self.test = pd.read_parquet(path_test) if path_test else None

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
        return (self.datasets_dir / dataset_name).exists()

    def read_dataset(self, filename: str) -> DataFrame:
        """
        Read a data file and return a DataFrame
        """
        file_path = self.datasets_dir.joinpath(filename)
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
