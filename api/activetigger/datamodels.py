import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any, Callable, Literal, Optional

from pandas import DataFrame
from pydantic import BaseModel, ConfigDict  # for dataframe
from sklearn.base import BaseEstimator  # type: ignore[import]

# Data model to use of the API


class ProjectBaseModel(BaseModel):
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
    cols_label: list[str] = []
    cols_context: list[str] = []
    test: bool = False
    n_total: int | None = None
    clear_test: bool = False
    random_selection: bool = False
    cols_stratify: list[str] = []
    stratify_train: bool = False
    stratify_test: bool = False
    force_label: bool = False
    force_computation: bool = False


class ProjectModel(ProjectBaseModel):
    """
    Once created
    """

    project_slug: str
    all_columns: list[str] | None = None


class ProjectDataModel(ProjectBaseModel):
    """
    To create a new project
    """

    csv: str


class AnnotationsDataModel(BaseModel):
    col_id: str
    col_label: str
    scheme: str
    csv: str
    filename: str | None = None


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
    selection: str = "fixed"
    sample: str = "untagged"
    label: str | None = None
    label_maxprob: str | None = None
    frame: list[Any] | None = None
    history: list[str] = []
    filter: str | None = None
    dataset: str = "train"


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
    selection: str | None = None


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
    kind: str = "multiclass"
    labels: list[str] = []


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

    model: str
    features: list
    params: (
        dict[str, str | float | bool | int | None]
        | dict[str, dict[str, str | float | bool | int | None]]
        | None
    )
    scheme: str
    standardize: bool | None = True
    dichotomize: str | None = None
    cv10: bool = False


class LMParametersModel(BaseModel):
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


class LMParametersModelTrained(LMParametersModel):
    """
    Parameters for bertmodel once trained
    """

    base_model: str
    n_train: int
    test_size: float


class LMParametersDbModel(LMParametersModel):
    predicted: bool = False
    compressed: bool = False


class BertModelModel(BaseModel):
    """
    Request Bertmodel
    TODO : model for parameters
    """

    project_slug: str
    scheme: str
    name: str
    base_model: str
    params: LMParametersModel
    test_size: float = 0.2
    dichotomize: str | None = None
    class_min_freq: int = 1
    class_balance: bool = False
    exclude_labels: list[str] = []


class UmapModel(BaseModel):
    """
    Params UmapModel
    """

    n_neighbors: int
    n_components: int
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


class ProjectionParametersModel(BaseModel):
    """
    Request projection
    """

    method: str
    features: list
    parameters: dict[str, float | str | bool | list] = {}


class ProjectionDataModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    data: DataFrame
    parameters: ProjectionParametersModel


class ProjectionOutModel(BaseModel):
    """
    Posting projection
    """

    status: str
    index: list
    x: list
    y: list
    parameters: ProjectionParametersModel
    labels: list[str] | None = None
    predictions: list[str] | None = None


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
    class_prior: str | None = None


class GenerationCreationModel(BaseModel):
    """
    GenAI model used in generation
    """

    slug: str
    api: str
    name: str
    endpoint: str | None = None
    credentials: str | None = None


class GenerationModel(GenerationCreationModel):
    """
    GenAI model used in generation
    """

    id: int


class GenerationAvailableModel(BaseModel):
    """
    GenAI models available for generation
    """

    slug: str
    api: str
    name: str


class GenerationModelApi(BaseModel):
    """
    GenAI API available for generation
    """

    name: str
    models: list[GenerationAvailableModel]


class MLStatisticsModel(BaseModel):
    f1_label: dict[str, float] | None = None
    precision_label: dict[str, float] | None = None
    recall_label: dict[str, float] | None = None
    f1_weighted: float | None = None
    f1_micro: float | None = None
    f1_macro: float | None = None
    accuracy: float | dict[str, float] | None = None
    precision: float | dict[str, float] | None = None
    confusion_matrix: list[list[int]] | None = None
    false_predictions: dict[str, Any] | list[Any] | None = None
    table: dict[str, Any] | None = None


class GenerationRequest(BaseModel):
    """
    To start a generating prompt
    """

    model_id: int
    token: str | None = None
    prompt: str
    n_batch: int = 1
    scheme: str
    mode: str = "all"


# --------------------
# CLASS FOR COMPUTING
# --------------------


class ProcessComputing(BaseModel):
    user: str
    unique_id: str
    time: datetime.datetime
    kind: str


class LMComputing(ProcessComputing):
    model_name: str
    status: Literal["training", "testing", "predicting"]
    scheme: Optional[str] = None
    dataset: Optional[str] = None
    get_progress: Callable[[], float | None] | None = None
    params: dict[str, Any] | None = None


class ProjectionComputing(ProcessComputing):
    kind: Literal["projection"]
    name: str
    method: str
    params: ProjectionParametersModel


class FeatureComputing(ProcessComputing):
    kind: Literal["feature"]
    name: str
    type: str
    parameters: dict


class GenerationComputing(ProcessComputing):
    kind: Literal["generation"]
    project: str
    number: int
    model_id: int
    get_progress: Callable[[], float | None] | None = None


class SimpleModelComputing(ProcessComputing):
    """
    Simplemodel object
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    name: str
    user: str
    features: list
    scheme: str
    labels: list
    model_params: dict
    standardize: bool
    model: BaseEstimator
    proba: DataFrame | None = None
    cv10: bool = False
    statistics: MLStatisticsModel | None = None
    statistics_cv10: MLStatisticsModel | None = None


class GenerationComputingOut(BaseModel):
    """
    Response for generation
    """

    model_id: int
    progress: float | None


class TableOutModel(BaseModel):
    """
    Response for table of elements
    """

    items: list
    total: int | float


class TableInModel(BaseModel):
    """
    Requesting a table of elements
    """

    list_ids: list
    list_labels: list
    scheme: str
    action: str


class TableBatchInModel(BaseModel):
    """
    Requesting a batch of elements
    """

    scheme: str
    min: int = 0
    max: int = 0
    mode: str = "all"
    contains: str | None = None
    dataset: str = "train"


class ProjectsServerModel(BaseModel):
    """
    Response for available projects
    """

    projects: list[str]
    auth: list


class ProjectSummaryModel(BaseModel):
    project_slug: str
    parameters: ProjectModel
    user_right: str
    created_by: str
    created_at: str
    size: float | None = None
    last_activity: str | None = None


class AvailableProjectsModel(BaseModel):
    """
    Response for available projects
    """

    projects: list[ProjectSummaryModel]


## State definition of the project


class NextProjectStateModel(BaseModel):
    methods_min: list[str]
    methods: list[str]
    sample: list[str]


class SchemesProjectStateModel(BaseModel):
    available: dict[str, SchemeModel]


class FeaturesProjectStateModel(BaseModel):
    options: dict[str, dict[str, Any]]
    available: list[str]
    training: dict[str, dict[str, str | None]]


class SimpleModelsProjectStateModel(BaseModel):
    options: dict[str, Any]
    available: dict[str, dict[str, dict[str, Any]]]
    training: dict[str, list[str]]


class LanguageModelsProjectStateModel(BaseModel):
    options: list[dict[str, Any]]
    available: dict[str, dict[str, dict[str, bool]]]
    training: dict[str, dict[str, str | float | int | dict | None]]
    base_parameters: LMParametersModel


class ProjectionsProjectStateModel(BaseModel):
    options: dict[str, dict[str, Any]]
    available: dict[str, str | int]
    training: dict[str, str]


class GenerationsProjectStateModel(BaseModel):
    training: dict[str, GenerationComputingOut]


class ErrorsProjectStateModel(BaseModel):
    errors: list[list]


class ProjectStateModel(BaseModel):
    """
    Response for server state
    """

    params: ProjectModel
    next: NextProjectStateModel
    schemes: SchemesProjectStateModel
    features: FeaturesProjectStateModel
    simplemodel: SimpleModelsProjectStateModel
    languagemodels: LanguageModelsProjectStateModel
    projections: ProjectionsProjectStateModel
    generations: GenerationsProjectStateModel
    errors: list[list]
    memory: float | None = None
    last_activity: str | None = None


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

    auth: dict[str, str]


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
    batch: DataFrame
    total: int
    min: int
    max: int
    filter: str | None

    class Config:
        arbitrary_types_allowed: bool = True  # Allow DataFrame type but switches off Pydantic here


class CodebookModel(BaseModel):
    content: str
    scheme: str
    time: str


class GenerationResult(BaseModel):
    user: str
    project_slug: str
    model_id: int
    element_id: str
    prompt: str
    answer: str


class GpuInformationModel(BaseModel):
    gpu_available: bool
    total_memory: float
    available_memory: float


class ServerStateModel(BaseModel):
    version: str
    queue: dict[str, dict[str, str | None]]
    active_projects: dict[str, dict[str, Any]]
    gpu: GpuInformationModel
    cpu: dict
    memory: dict
    disk: dict


class StaticFileModel(BaseModel):
    name: str
    path: str


class ProjectStaticFiles(BaseModel):
    dataset: StaticFileModel
    model: StaticFileModel | None = None


class FeatureDescriptionModel(BaseModel):
    name: str
    parameters: dict[str, Any]
    user: str
    time: str
    kind: str
    cols: list[str]


class FitModelResults(BaseModel):
    model: Any
    proba: DataFrame
    statistics: MLStatisticsModel
    statistics_cv10: MLStatisticsModel | None = None
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SimpleModelOutModel(BaseModel):
    """
    Trained simplemodel
    """

    features: list
    model: str
    params: (
        dict[str, str | float | bool | list | None]
        | dict[str, dict[str, str | float | bool | None]]
        | None
    )
    scheme: str
    username: str
    statistics: MLStatisticsModel
    statistics_cv10: MLStatisticsModel | None = None


class ReturnTaskPredictModel(BaseModel):
    path: str
    metrics: MLStatisticsModel | None = None


class LMInformationsModel(BaseModel):
    params: dict | None = None
    loss: dict | None = None
    train_scores: dict | None = None
    test_scores: dict | None = None
    valid_scores: dict | None = None
    outofsample_scores: dict | None = None


class ProjectUpdateModel(BaseModel):
    project_name: str | None = None
    language: str | None = None
    cols_text: list[str] | None = None
    cols_context: list[str] | None = None
    add_n_train: int | None = None


class UserStatistics(BaseModel):
    username: str
    projects: dict[str, str]
    # last_connexion
    # last_actions


class PromptInputModel(BaseModel):
    text: str
    name: str | None = None


class PromptModel(BaseModel):
    id: int
    text: str
    parameters: dict[str, Any]


class TextDatasetModel(BaseModel):
    id: str
    text: str
    filename: str | None = None
    csv: str | None = None


class GeneratedElementsIn(BaseModel):
    n_elements: int
    filters: list[str] = []


class ExportGenerationsParams(BaseModel):
    filters: list[str] = []


class LanguageModelScheme(BaseModel):
    name: str
    scheme: str
    parameters: dict[str, Any]
    path: str
