from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# Data model to use of the API


class BaseProjectModel(BaseModel):
    """
    Parameters of a project to save in the database
    """

    cols_text: list[str]
    project_name: str
    filename: str
    col_id: str
    n_train: int
    n_test: int
    dir: Path | None = None
    embeddings: list[str] = []
    n_skip: int = 0
    default_scheme: list[str] = []
    language: str = "fr"
    col_label: str | None = None
    cols_context: list[str] = []
    cols_test: list[str] = []
    test: bool = False
    n_total: int | None = None
    clear_test: bool = False
    random_selection: bool = False


class ProjectModel(BaseProjectModel):
    """
    Once created
    """

    project_slug: str
    all_columns: list[str] | None = None


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
    col_label: str | None = None
    scheme: str | None = None


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
    label: str | None = None
    frame: list[Any] | None = None
    history: list[str] = []
    filter: str | None = None


class ElementOutModel(BaseModel):
    """
    Posting element to annotate
    """

    element_id: str
    text: str
    context: dict[str, Any]
    selection: str
    info: str | None
    predict: dict[str, Any]
    frame: list | None
    limit: int | None
    history: list | None = None
    n_sample: int | None = None


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
    list of users on the server
    """

    users: dict[str, dict[str, str]]
    auth: list[str]


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
    label: str | None
    dataset: str | None = "train"
    comment: str | None = None


class TableAnnotationsModel(BaseModel):
    """
    Table of annotations
    """

    annotations: list[AnnotationModel]
    dataset: str | None = "train"


class SchemeModel(BaseModel):
    """
    Specific scheme
    """

    project_slug: str
    name: str
    kind: str | None = "multiclass"
    labels: list[str] | None = None


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
    params: (
        dict[str, str | float | bool | int | None]
        | dict[str, dict[str, str | float | bool | int | None]]
        | None
    )
    # TODO CAN BE BETTER
    scheme: str
    standardize: bool | None = True
    dichotomize: str | None = None


class SimpleModelOutModel(BaseModel):
    """
    Trained simplemodel
    """

    features: list
    model: str
    params: (
        dict[str, str | float | bool | None]
        | dict[str, dict[str, str | float | bool | None]]
        | None
    )
    scheme: str
    username: str
    statistics: dict


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
    scheme: str
    name: str
    base_model: str
    params: BertModelParametersModel
    test_size: float = 0.2
    dichotomize: str | None = None
    class_min_freq: int = 1
    class_balance: bool = False


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
    features: list
    params: dict[str, Any]


class ProjectionInStrictModel(BaseModel):
    """
    Request projection
    """

    method: str
    features: list
    params: TsneModel | UmapModel


class ProjectionOutModel(BaseModel):
    """
    Posting projection
    """

    status: str
    index: list
    x: list
    y: list
    labels: list


#    texts: list


class FeatureModel(BaseModel):
    """
    Feature model
    """

    type: str
    name: str
    parameters: dict[str, str | float]


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
    fit_prior: bool = True
    class_prior: str | None | None = None


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
    token: str | None = None
    prompt: str
    n_batch: int = 1
    scheme: str
    mode: str = "all"


class TableOutModel(BaseModel):
    """
    Response for table of elements
    """

    items: list
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

    projects: list[str]
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

    projects: list[ProjectSummaryModel]


class ProjectStateModel(BaseModel):
    """
    Response for server state
    TODO : have a more precise description of the fields
    """

    params: ProjectModel
    users: dict[str, Any]
    next: dict[str, Any]
    schemes: dict[str, Any]
    features: dict[str, Any]
    simplemodel: dict[str, Any]
    bertmodels: dict[str, Any]
    projections: dict[str, Any]
    generations: dict[str, Any]
    errors: list[list]


class QueueModel(BaseModel):
    """
    Response for current queue
    """

    content: dict[str, dict[str, Any]]


class ProjectDescriptionModel(BaseModel):
    """
    Project description
    """

    users: list[str]
    train_set_n: int
    train_annotated_n: int
    train_annotated_distribution: dict[str, Any]
    test_set_n: int | None = None
    test_annotated_n: int | None = None
    test_annotated_distribution: dict[str, Any] | None = None
    sm_10cv: Any | None = None


class ProjectAuthsModel(BaseModel):
    """
    Auth description for a project
    """

    auth: dict[str, Any]


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

    credits: list[str]
    page: str
    documentation: str
    contact: str


class ReconciliationModel(BaseModel):
    """
    list of elements to reconciliate
    """

    table: list[dict[str, str | dict[str, str]]]
    users: list[str]


class AuthActions(StrEnum):
    add = "add"
    delete = "delete"


class TableBatch(BaseModel):
    batch: Any
    total: int
    min: int
    max: int
    filter: str


class CodebookModel(BaseModel):
    content: str
    scheme: str
    time: str | None = None
