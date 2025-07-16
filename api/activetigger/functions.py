import io
import os
import string
import unicodedata
from getpass import getpass
from typing import Any, cast
from urllib.parse import quote

import bcrypt
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
from transformers import (  # type: ignore[import]
    BertTokenizer,
)

from activetigger.datamodels import GpuInformationModel, MLStatisticsModel


def slugify(text: str, way: str = "file") -> str:
    """
    Convert a string to a slug format
    """
    if way == "file":
        return python_slugify(text)
    elif way == "url":
        return quote(text)
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


def truncate_text(text: str, max_tokens: int = 512):
    """
    Limit a text to a specific number of tokens
    """
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(text)
    num_tokens = len(tokens)
    if num_tokens > max_tokens:
        print(num_tokens)
        truncated_tokens = tokens[:max_tokens]
        text_t = tokenizer.convert_tokens_to_string(truncated_tokens)
    else:
        text_t = text
    return text_t


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
    Y_true: Series, Y_pred: Series, texts: Series | None = None, decimals: int = 3
) -> MLStatisticsModel:
    """
    Compute metrics for a prediction
    """
    labels = list(Y_true.unique())
    precision = [
        round(i, decimals)
        for i in precision_score(
            list(Y_true),
            list(Y_pred),
            average=None,
            labels=labels,
            zero_division=1,
        )
    ]
    f1 = [
        round(i, decimals)
        for i in f1_score(list(Y_true), list(Y_pred), average=None, labels=labels)
    ]
    confusion = confusion_matrix(Y_true, Y_pred, labels=labels)

    recall = [
        round(i, decimals)
        for i in recall_score(list(Y_true), list(Y_pred), average=None, labels=labels)
    ]

    table = pd.DataFrame(confusion, index=labels, columns=labels)
    table["Total"] = table.sum(axis=1)
    table = table.T
    table["Total"] = table.sum(axis=1)
    table = table.T

    # Create a table of false predictions
    filter_false_prediction = Y_true != Y_pred
    if texts is not None:
        tab = pd.concat(
            [Y_true[filter_false_prediction], Y_pred[filter_false_prediction], texts],
            axis=1,
            join="inner",
        ).reset_index()
        tab.columns = pd.Index(["id", "label", "prediction", "text"])
        false_prediction = tab.to_dict(orient="records")
    else:
        false_prediction = filter_false_prediction.loc[lambda x: x].index.tolist()

    statistics = MLStatisticsModel(
        f1_label=dict(
            zip(
                labels,
                f1,
            )
        ),
        f1_weighted=round(f1_score(Y_true, Y_pred, average="weighted"), decimals),
        f1_macro=round(f1_score(Y_true, Y_pred, average="macro"), decimals),
        f1_micro=round(f1_score(Y_true, Y_pred, average="micro"), decimals),
        accuracy=round(accuracy_score(Y_true, Y_pred), decimals),
        precision=round(
            precision_score(list(Y_true), list(Y_pred), average="micro", zero_division=1),
            decimals,
        ),
        precision_label=dict(
            zip(
                labels,
                precision,
            )
        ),
        recall_label=dict(
            zip(
                labels,
                recall,
            )
        ),
        confusion_matrix=confusion.tolist(),
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
