import os
from collections.abc import Callable
from enum import StrEnum

from dotenv import load_dotenv


#  Singleton utils
class _Singleton(type):
    _instances: dict = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# load .env variables if exists
load_dotenv()


# Enum for env mode
class MODE(StrEnum):
    DEV = "dev"
    PROD = "prod"

    @classmethod
    def has_member_key(cls, key):
        return key in cls.__members__


# utils to cast str env variables as int or float
def parse_environ(key: str, parse_method: Callable[[str], int | float], default: int | float):
    if key is None:
        raise ValueError(f"Key {key} is None")
    try:
        return parse_method(os.environ.get(key))  # type: ignore
    except Exception:
        return default


class Config(metaclass=_Singleton):
    # type sage configuration specification with default values coming from env variables or defaults
    data_path: str = os.environ.get("DATA_PATH", ".")
    user_hdd_max: float
    mode: MODE = os.environ.get("MODE", str(MODE.DEV))  # type: ignore
    secret_key: str = os.environ.get("SECRET_KEY", "Q__zz0ew00R_YSwCFl-6VgS9dPbfDtFDnzHfd57t0EY=")
    database_url: str
    root_password: str | None = os.environ.get("ROOT_PASSWORD", None)
    # orchestrator
    jwt_algorithm: str = os.environ.get("JWT_ALGORITHM", "HS256")
    max_loaded_projects: int
    n_workers_gpu: int
    n_workers_cpu: int
    update_timeout: int

    def __init__(self):
        # for variables which needs cast or other treatment we do that work in the constructor
        self.mode = (
            os.environ.get("MODE")
            if os.environ.get("MODE") is not None and MODE.has_member_key(os.environ.get("MODE"))
            else "dev"
        )
        self.user_hdd_max = parse_environ("ACTIVETIGGER_USER_HDD_MAX", float, 30.0)
        self.max_loaded_projects = parse_environ("MAX_LOADED_PROJECTS", int, 20)
        self.n_workers_gpu = parse_environ("N_WORKERS_GPU", int, 1)
        self.n_workers_cpu = parse_environ("N_WORKERS_CPU", int, 5)
        self.update_timeout = parse_environ("UPDATE_TIMEOUT", int, 1)
        self.database_url = os.environ.get(
            "DATABASE_URL",
            f"sqlite:///{os.path.join(self.data_path, 'projects', 'activetigger.db')}",
        )


# the configuration is safe to share as it's a singleton (initialized only once)
config = Config()
