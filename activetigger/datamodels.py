from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from typing import Optional, List, Dict, Any
from enum import StrEnum

# Data model to use of the API


class ProjectModel(BaseModel):
    """
    Parameters of a project to save in the database
    """

    project_name: str
    filename: str
    col_text: str
    col_id: str
    n_train: int
    n_test: int
    dir: Path | None = None
    embeddings: list = []
    n_skip: int = 0
    default_scheme: list = []
    language: str = "fr"
    col_label: str | None = None
    cols_context: list = []
    cols_test: list = []
    test: bool = False


class ProjectDataModel(ProjectModel):
    """
    Parameters of a project to create it
    (with data field)
    """

    csv: str


class ActionModel(str, Enum):
    """
    Type of actions available
    """

    delete = "delete"
    add = "add"
    update = "update"


class NextInModel(BaseModel):
    """
    Requesting next element to annotate
    """

    scheme: str
    selection: str = "deterministic"
    sample: str = "untagged"
    tag: str | None = None
    frame: list[float] | None = None
    history: list = []


class ElementOutModel(BaseModel):
    """
    Posting element to annotate
    """

    element_id: str
    text: str
    context: Dict[str, Any]
    selection: str
    info: str | None
    predict: dict
    frame: list | None
    limit: int | None


class UserModel(BaseModel):
    """
    User definition
    """

    username: str
    status: str | None


class UserInDBModel(UserModel):
    """
    Adding password to user definition
    """

    hashed_password: str


class UsersServerModel(BaseModel):
    """
    List of users on the server
    """

    users: list
    auth: list


class TokenModel(BaseModel):
    """
    Auth token
    """

    access_token: str
    token_type: str
    status: str | None


class AnnotationModel(BaseModel):
    """
    Specific Annotatoin
    """

    project_name: str
    element_id: str
    tag: str
    user: str
    scheme: str
    selection: Optional[str] = None


class SchemeModel(BaseModel):
    """
    Specific scheme
    """

    project_name: str
    name: str
    tags: Optional[list] = []


class RegexModel(BaseModel):
    """
    Regex
    """

    project_name: str
    name: str
    value: str
    user: str


class SimpleModelModel(BaseModel):
    """
    Request Simplemodel
    TODO : model for parameters
    """

    features: list
    model: str
    params: dict | None
    scheme: str
    standardize: Optional[bool] = True


class BertModelModel(BaseModel):
    """
    Request Bertmodel
    TODO : model for parameters
    """

    project_name: str
    user: str
    scheme: str
    name: str
    base_model: str
    params: dict
    test_size: float


class ProjectionInModel(BaseModel):
    """
    Request projection
    TODO : model for parameters
    """

    method: str
    features: list
    params: dict


class ProjectionOutModel(BaseModel):
    """
    Posting projection
    """

    status: str
    index: List
    x: List
    y: List
    labels: List
    texts: List


class FeatureModel(BaseModel):
    params: dict


class LiblinearParams(BaseModel):
    cost: float


class KnnParams(BaseModel):
    n_neighbors: int


class RandomforestParams(BaseModel):
    n_estimators: int
    max_features: int | None


class LassoParams(BaseModel):
    C: int


class Multi_naivebayesParams(BaseModel):
    alpha: float
    fit_prior: bool
    class_prior: bool


class BertParams(BaseModel):
    batchsize: int
    gradacc: float
    epochs: int
    lrate: float
    wdecay: float
    best: bool
    eval: int
    adapt: bool


class UmapModel(BaseModel):
    n_neighbors: int
    min_dist: float
    n_components: int
    metric: str


class TsneModel(BaseModel):
    n_components: int
    learning_rate: str | float
    init: str
    perplexity: int


class ZeroShotModel(BaseModel):
    scheme: str
    prompt: str
    api: str
    token: str
    number: int = 10


class TableOutModel(BaseModel):
    """
    Response for table of elements
    """

    id: List[str]
    timestamp: List[str]
    label: List[str]
    text: List[str]


class TableLogsModel(BaseModel):
    """
    Response for table of logs
    """

    time: List
    user: List
    project: List
    action: List


class TableInModel(BaseModel):
    """
    Requesting a table of elements
    """

    list_ids: list
    list_labels: list
    scheme: str
    action: str


class ProjectsServerModel(BaseModel):
    """
    Response for available projects
    """

    projects: List[str]
    auth: list


class AvailableProjectsModel(BaseModel):
    """
    Response for available projects
    """

    projects: Dict[str, Dict[str, Any]]


class StateModel(BaseModel):
    """
    Response for server state
    TODO : have a more precise description of the fields
    """

    params: ProjectModel
    next: Dict[str, Any]
    schemes: Dict[str, Any]
    features: Dict[str, Any]
    simplemodel: Dict[str, Any]
    bertmodels: Dict[str, Any]
    projections: Dict[str, Any]
    zeroshot: Dict[str, Any]


class QueueModel(BaseModel):
    """
    Response for current queue
    """

    content: Dict[str, Dict[str, Any]]


class ProjectDescriptionModel(BaseModel):
    """
    Project description
    """

    trainset_n: int
    annotated_n: int
    users: List[str]
    annotated_distribution: Dict[str, Any]
    testset_n: Optional[int] = None
    sm_10cv: Optional[Any] = None


class ProjectAuthsModel(BaseModel):
    """
    Auth description for a project
    """

    auth: Dict[str, Any]


class WaitingModel(BaseModel):
    """
    Response for waiting
    """

    detail: str
    status: str = "waiting"


class DocumentationModel(BaseModel):
    """
    Documentation model
    """

    credits: List[str]
    page: str
    documentation: str
    contact: str


class ReconciliationModel(BaseModel):
    """
    List of elements to reconciliate
    """

    list_disagreements: List[Dict[str, Any]]


class AuthActions(StrEnum):
    add = "add"
    delete = "delete"
