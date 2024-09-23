from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

# Data model to use of the API


class BaseProjectModel(BaseModel):
    """
    Parameters of a project to save in the database
    """

    project_name: str
    filename: str
    col_text: str | List[str]
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


class ProjectModel(BaseProjectModel):
    """
    Once created
    """

    project_slug: str


class ProjectDataModel(BaseProjectModel):
    """
    To create a new project
    """

    csv: str


class TestSetDataModel(BaseModel):
    col_text: str
    col_id: str
    n_test: int
    filename: str
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
    frame: list[Any] | None = None
    history: list = []
    filter: Optional[str] = None


class ElementOutModel(BaseModel):
    """
    Posting element to annotate
    """

    element_id: str
    text: str
    context: Dict[str, Any]
    selection: str
    info: str | None
    predict: Dict[str, Any]
    frame: list | None
    limit: int | None
    history: list


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

    users: List[str]
    auth: List[str]


class TokenModel(BaseModel):
    """
    Auth token
    """

    access_token: str
    token_type: str
    status: str | None


class AnnotationModel(BaseModel):
    """
    Specific Annotation
    """

    project_slug: str
    scheme: str
    element_id: str
    label: str
    dataset: Optional[str] = "train"


class TableAnnotationsModel(BaseModel):
    """
    Table of annotations
    """

    annotations: List[AnnotationModel]
    dataset: Optional[str] = "train"


class SchemeModel(BaseModel):
    """
    Specific scheme
    """

    project_slug: str
    name: str
    tags: Optional[list] = []


class RegexModel(BaseModel):
    """
    Regex
    """

    project_slug: str
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
    params: Dict[str, Any] | Dict[str, Dict[str, Any]] | None
    scheme: str
    standardize: Optional[bool] = True


class BertModelParametersModel(BaseModel):
    """
    Parameters for bertmodel training
    """

    batchsize: int = 4
    gradacc: float = 1
    epochs: int = 3
    lrate: float = 5e-05
    wdecay: float = 0.01
    best: bool = True
    eval: int = 10
    gpu: bool = False
    adapt: bool = True


class BertModelModel(BaseModel):
    """
    Request Bertmodel
    TODO : model for parameters
    """

    project_slug: str
    # user: str
    scheme: str
    name: str
    base_model: str
    params: dict | BertModelParametersModel
    test_size: float


class UmapModel(BaseModel):
    """
    Params UmapModel
    """

    n_components: int
    n_neighbors: int
    min_dist: float
    metric: str


class TsneModel(BaseModel):
    """
    Params TsneModel
    """

    n_components: int
    learning_rate: str | float
    init: str
    perplexity: int


class ProjectionInModel(BaseModel):
    """
    Request projection
    """

    method: str
    features: List
    params: Dict[str, Any]


class ProjectionInStrictModel(BaseModel):
    """
    Request projection
    """

    method: str
    features: List
    params: TsneModel | UmapModel


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
    """
    Feature model
    """

    type: str
    name: str
    parameters: Dict[str, Union[str, float]]


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


class GenerateModel(BaseModel):
    """
    To start a generating prompt
    """

    api: str
    endpoint: str
    token: Optional[str] = None
    prompt: str
    n_batch: int = 1
    scheme: str
    mode: str = "all"


class TableOutModel(BaseModel):
    """
    Response for table of elements
    """

    items: List
    total: int


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


class ProjectSummaryModel(BaseModel):
    parameters: ProjectModel
    user_right: str
    created_by: str
    created_at: str


class AvailableProjectsModel(BaseModel):
    """
    Response for available projects
    """

    projects: List[ProjectSummaryModel]


class ProjectStateModel(BaseModel):
    """
    Response for server state
    TODO : have a more precise description of the fields
    """

    params: ProjectModel
    users: Dict[str, Any]
    next: Dict[str, Any]
    schemes: Dict[str, Any]
    features: Dict[str, Any]
    simplemodel: Dict[str, Any]
    bertmodels: Dict[str, Any]
    projections: Dict[str, Any]
    generations: Dict[str, Any]


class QueueModel(BaseModel):
    """
    Response for current queue
    """

    content: Dict[str, Dict[str, Any]]


class ProjectDescriptionModel(BaseModel):
    """
    Project description
    """

    users: List[str]
    train_set_n: int
    train_annotated_n: int
    train_annotated_distribution: Dict[str, Any]
    test_set_n: Optional[int] = None
    test_annotated_n: Optional[int] = None
    test_annotated_distribution: Optional[Dict[str, Any]] = None
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

    table: List[Dict[str, str | Dict[str, str]]]
    users: List[str]


class AuthActions(StrEnum):
    add = "add"
    delete = "delete"
