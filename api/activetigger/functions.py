import io
import os
from getpass import getpass
from typing import cast

import bcrypt
import pandas as pd
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
from transformers import (  # type: ignore[import]
    BertTokenizer,
)

from activetigger.datamodels import MLStatisticsModel


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


def compare_to_hash(text: str, hash: str | bytes):
    """
    Compare string to its hash
    """

    bytes_hash: bytes
    if type(hash) is str:
        bytes_hash = hash.encode()
    else:
        bytes_hash = cast(bytes, hash)
    r = bcrypt.checkpw(text.encode(), bytes_hash)
    return r


def tokenize(texts: Series, language: str = "fr") -> Series:
    """
    Clean texts with tokenization to facilitate word count
    TODO : faster tokenization ?
    """
    if language == "en":
        model = "en_core_web_sm"
    elif language == "fr":
        model = "fr_core_news_sm"
    else:
        raise Exception(f"Language {language} is not supported")

    nlp = spacy.load(model, disable=["ner", "tagger"])
    docs = nlp.pipe(texts, batch_size=1000)
    textes_tk = [" ".join([str(token) for token in doc]) for doc in docs]
    return pd.Series(textes_tk, index=texts.index)


def get_gpu_memory_info() -> dict:
    """
    Get info on GPU
    """
    if not torch.cuda.is_available():
        return {"gpu_available": False, "total_memory": 0, "available_memory": 0}

    torch.cuda.empty_cache()
    mem = torch.cuda.mem_get_info()

    return {
        "gpu_available": True,
        "total_memory": round(mem[1] / 1e9, 2),  # Convert to GB
        "available_memory": round(mem[0] / 1e9, 2),  # Convert to GB
    }


def get_gpu_estimate():
    return None


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


def clean_regex(text: str):
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
    print(secret_key)
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


def get_metrics(Y_true: Series, Y_pred: Series, decimals: int = 3) -> MLStatisticsModel:
    """
    Compute metrics for a prediction
    """
    labels = list(Y_true.unique())
    statistics = MLStatisticsModel(
        f1_label=dict(
            zip(
                labels,
                [
                    round(i, decimals)
                    for i in f1_score(
                        list(Y_true), list(Y_pred), average=None, labels=labels
                    )
                ],
            )
        ),
        f1_weighted=round(f1_score(Y_true, Y_pred, average="weighted"), decimals),
        f1_macro=round(f1_score(Y_true, Y_pred, average="macro"), decimals),
        f1_micro=round(f1_score(Y_true, Y_pred, average="micro"), decimals),
        accuracy=round(accuracy_score(Y_true, Y_pred), decimals),
        precision=round(
            precision_score(
                list(Y_true),
                list(Y_pred),
                average="micro",
            ),
            decimals,
        ),
        precision_label=dict(
            zip(
                labels,
                [
                    round(i, decimals)
                    for i in precision_score(
                        list(Y_true), list(Y_pred), average=None, labels=labels
                    )
                ],
            )
        ),
        recall_label=dict(
            zip(
                labels,
                [
                    round(i, decimals)
                    for i in recall_score(
                        list(Y_true), list(Y_pred), average=None, labels=labels
                    )
                ],
            )
        ),
        confusion_matrix=confusion_matrix(Y_true, Y_pred).tolist(),
        false_predictions=(Y_true != Y_pred).loc[lambda x: x].index.tolist(),
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


def process_payload_csv(csv_str: str, cols: list[str]):
    """
    Process payload from a CSV file in str
    """
    csv_buffer = io.StringIO(csv_str)
    df = pd.read_csv(
        csv_buffer,
    )
    return df[cols]
