import multiprocessing
from logging import Logger
from pathlib import Path
from typing import Optional, cast
from getpass import getpass

import bcrypt
import pandas as pd
import spacy
import torch
from cryptography.fernet import Fernet
from pandas import Series
from sklearn.preprocessing import OneHotEncoder
from transformers import (
    BertTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class CustomLoggingCallback(TrainerCallback):
    event: Optional[multiprocessing.synchronize.Event]
    current_path: Path
    logger: Logger

    def __init__(self, event, logger, current_path):
        self.event = event
        self.current_path = current_path
        self.logger = logger

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.logger.info(f"Step {state.global_step}")
        progress_percentage = (state.global_step / state.max_steps) * 100
        with open(self.current_path.joinpath("train/progress"), "w") as f:
            f.write(str(progress_percentage))
        # end if event set
        if self.event is not None:
            if self.event.is_set():
                self.logger.info("Event set, stopping training.")
                control.should_training_stop = True
                raise Exception("Process interrupted by user")


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
