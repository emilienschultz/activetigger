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

    train: DataFrame
    valid: DataFrame | None
    test: DataFrame | None

    def __init__(self, path_train: Path, path_valid: Path | None, path_test: Path | None):
        if not path_train.exists():
            raise FileNotFoundError(f"Training data file not found: {path_train}")
        if path_valid and not path_valid.exists():
            raise FileNotFoundError(f"Validation data file not found: {path_valid}")
        if path_test and not path_test.exists():
            raise FileNotFoundError(f"Test data file not found: {path_test}")

        self.train = pd.read_parquet(path_train)
        self.valid = pd.read_parquet(path_valid) if path_valid else None
        self.test = pd.read_parquet(path_test) if path_test else None
