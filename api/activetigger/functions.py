import io
import json
import os
import string
import unicodedata
from getpass import getpass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import bcrypt
import numpy as np
import pandas as pd  # type: ignore[import]
import spacy
import torch
from cryptography.fernet import Fernet
from pandas import Series
from sklearn.metrics import (  # type: ignore[import]
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import OneHotEncoder  # type: ignore[import]
from slugify import slugify as python_slugify  # type: ignore[import]

from activetigger.datamodels import GpuInformationModel, MLStatisticsModel


def slugify(text: str, way: str = "file") -> str:
    """
    Convert a string to a slug format
    """
    if way == "file":
        return python_slugify(text)
    elif way == "url":
        return quote(text, safe="")
    else:
        raise ValueError("Invalid way parameter. Use 'file' or 'url'.")


def remove_punctuation(text) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def replace_accented_chars(text):
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def get_root_pwd() -> str:
    """
    Function to get the password in the CLI start
    """
    print("╔═════════════════════════════════╗")
    print("║    Define a Root Password       ║")
    print("╠═════════════════════════════════╣")
    print("║  Your password must be at least ║")
    print("║  6 characters long and entered  ║")
    print("║  twice to confirm.              ║")
    print("╚═════════════════════════════════╝")
    while True:
        root_password = getpass("Enter a root password : ")

        if len(root_password) < 6:
            print("The password need to have 6 character at minimum")
            continue
        confirm_password = getpass("Re-enter the root password: ")

        if root_password != confirm_password:
            print("Error: The passwords do not match. Please try again.")

        else:
            print("Password confirmed successfully.")
            print("Creating the entry in the database...")
            return root_password


def get_hash(text: str) -> bytes:
    """
    Build a hash string from text
    """
    salt = bcrypt.gensalt()
    hashed: bytes = bcrypt.hashpw(text.encode(), salt)
    return hashed


def compare_to_hash(text: str, hash: str | bytes) -> bool:
    """
    Compare string to its hash
    """

    bytes_hash: bytes
    if type(hash) is str:
        bytes_hash = hash.encode()
    else:
        bytes_hash = cast(bytes, hash)
    return bcrypt.checkpw(text.encode(), bytes_hash)


def tokenize(texts: Series, language: str = "fr", batch_size=100) -> Series:
    """
    Clean texts with tokenization to facilitate word count
    """

    models = {
        "en": "en_core_web_sm",
        "fr": "fr_core_news_sm",
        "de": "de_core_news_sm",
        "ja": "ja_core_news_sm",
        "cn": "zh_core_web_sm",
        "es": "es_core_news_sm",
    }
    if language not in models:
        raise Exception(f"Language {language} is not supported")
    nlp = spacy.load(models[language], disable=["ner", "tagger"])
    docs = nlp.pipe(texts, batch_size=batch_size)
    textes_tk = [" ".join([str(token) for token in doc]) for doc in docs]
    del nlp
    return pd.Series(textes_tk, index=texts.index)


def get_gpu_memory_info() -> GpuInformationModel:
    """
    Get info on GPU
    """
    if not torch.cuda.is_available():
        return GpuInformationModel(
            gpu_available=False,
            total_memory=0.0,
            available_memory=0.0,
        )
    else:
        torch.cuda.empty_cache()
        mem = torch.cuda.mem_get_info()

        return GpuInformationModel(
            gpu_available=True,
            total_memory=round(mem[1] / 1e9, 2),  # Convert to GB
            available_memory=round(mem[0] / 1e9, 2),  # Convert to GB
        )


def cat2num(df):
    """
    Transform a categorical variable to numerics
    """
    df = pd.DataFrame(df)
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df)
    encoded = pd.DataFrame(encoded, index=df.index)
    encoded.columns = ["col" + str(i) for i in encoded.columns]
    return encoded


def clean_regex(text: str) -> str:
    """
    Remove special characters from a string
    """
    if text == "\\" or text == "\\\\":
        text = ""
    if len(text) > 1 and text[-1] == "\\":
        text = text[:-1]
    return text


def encrypt(text: str | None, secret_key: str | None) -> str:
    """
    Encrypt a string
    """
    if text is None or secret_key is None:
        raise Exception("Text or secret key is None")
    cipher = Fernet(secret_key)
    encrypted_token = cipher.encrypt(text.encode())
    return encrypted_token.decode()


def decrypt(text: str | None, secret_key: str | None) -> str:
    """
    Decrypt a string
    """
    if text is None or secret_key is None:
        raise Exception("Text or secret key is None")
    cipher = Fernet(secret_key)

    decrypted_token = cipher.decrypt(text.encode())
    return decrypted_token.decode()


def get_metrics(
    Y_true: pd.Series,
    Y_pred: pd.Series,
    labels: list[str] | None = None,
    texts: pd.Series | None = None,
    decimals: int = 3,
) -> MLStatisticsModel:
    """
    Compute metrics for a prediction
    - precision, f1, recall per label
    - f1 (weighted macro, micro) and precision micro
    - confusion matrix and table
    """
    if labels is None:
        labels = list(Y_true.unique())

    # Compute scores per label --- --- --- --- --- --- --- --- --- --- --- --- -
    precision_label = precision_score(Y_true, Y_pred, average=None, labels=labels, zero_division=1)
    precision_label = [round(score, decimals) for score in precision_label]

    f1_label = f1_score(Y_true, Y_pred, average=None, labels=labels)
    f1_label = [round(score, decimals) for score in f1_label]

    recall_label = recall_score(Y_true, Y_pred, average=None, labels=labels)
    recall_label = [round(score, decimals) for score in recall_label]

    # Compute score averaged (micro, macro, weighted) --- --- --- --- --- --- --
    f1_weighted = f1_score(Y_true, Y_pred, average="weighted")
    f1_weighted = round(f1_weighted, decimals)

    f1_macro = f1_score(Y_true, Y_pred, average="macro")
    f1_macro = round(f1_macro, decimals)

    f1_micro = f1_score(Y_true, Y_pred, average="micro")
    f1_micro = round(f1_micro, decimals)

    precision_micro = precision_score(Y_true, Y_pred, average="micro", zero_division=1)
    precision_micro = round(precision_micro, decimals)

    # Compute confiusion matrix --- --- --- --- --- --- --- --- --- --- --- --- -
    confusion = confusion_matrix(Y_true, Y_pred, labels=labels)

    table = pd.DataFrame(confusion, index=labels, columns=labels)
    table["Total"] = table.sum(axis=1)
    table = table.T
    table["Total"] = table.sum(axis=1)
    table = table.T

    # Create a table of false predictions --- --- --- --- --- --- --- --- --- --
    filter_false_prediction = Y_true != Y_pred
    if texts is not None:
        # Conca
        tab = pd.concat(
            [
                pd.Series(Y_true[filter_false_prediction]),
                pd.Series(Y_pred[filter_false_prediction]),
                pd.Series(texts),
            ],
            axis=1,
            join="inner",
        ).reset_index()
        tab.columns = pd.Index(["id", "label", "prediction", "text"])
        false_prediction = tab.to_dict(orient="records")
    else:
        # TODO: explicit or refactor
        false_prediction = filter_false_prediction.loc[lambda x: x].index.tolist()

    statistics = MLStatisticsModel(
        f1_label=dict(zip(labels, f1_label)),
        precision_label=dict(zip(labels, precision_label)),
        recall_label=dict(zip(labels, recall_label)),
        confusion_matrix=confusion.tolist(),
        f1_weighted=f1_weighted,
        f1_macro=f1_macro,
        f1_micro=f1_micro,
        precision=precision_micro,
        false_predictions=false_prediction,
        table=cast(dict[str, Any], table.to_dict(orient="split")),
    )
    return statistics


def get_dir_size(path: str = ".") -> float:
    """
    Get size of a directory in MB
    """
    total: float = 0
    with os.scandir(path) as it:
        for entry in it:
            if entry.is_file():
                total = total + entry.stat().st_size / (1024 * 1024)
            elif entry.is_dir():
                total += get_dir_size(entry.path)
    return total


def process_payload_csv(csv_str: str, cols: list[str]) -> pd.DataFrame:
    """
    Process payload from a CSV file in str to get a DataFrame with specific columns
    """
    csv_buffer = io.StringIO(csv_str)
    df = pd.read_csv(
        csv_buffer,
    )
    return df[cols]


def get_model_metrics(path_model: Path) -> dict | None:
    """
    Get the scores of the model for a dataset
    - training metrics
    - last computed metrics
    """
    if not path_model.exists():
        raise Exception(f"The folder {path_model} does not exist")

    # training metrics
    if not path_model.joinpath("metrics_training.json").exists():
        raise Exception(f"The file metrics_training.json does not exist in {path_model}")
    with open(
        path_model.joinpath("metrics_training.json"),
        "r",
    ) as f:
        scores = json.load(f)

    # computed metrics and concatenate
    files = sorted(
        [
            f.name
            for f in path_model.iterdir()
            if f.is_file() and f.name.startswith("metrics_predict_")
        ],
    )
    if len(files) > 0:
        last_stat_file = files[-1]
        with open(path_model.joinpath(last_stat_file), "r") as f:
            stats = json.load(f)
        scores = {**scores, **stats}

    return scores
