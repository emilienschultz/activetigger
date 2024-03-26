from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from typing import Optional

class ProjectModel(BaseModel):
    """
    Parameters of a project
    """
    project_name:str
    user:str
    col_text:str
    col_id:str
    n_rows:int = 2000
    n_train:int
    n_test:int
    dir:Path|None = None
    embeddings:list = []
    n_skip:int = 0
    default_scheme: list = []
    langage:str = "fr"
    col_label:str|None = None # TODO: load existing tags
    cols_context:list = [] # TODO: select variable to keep

class Action(str, Enum):
    delete = "delete"
    add = "add"
    update = "update"

class Scheme(BaseModel):
    """
    Set of labels
    """
    labels:list[str]

class NextModel(BaseModel):
    """
    Request of an element
    """
    scheme:str = "default"
    mode:str = "deterministic"
    on:str|None = "untagged"

class SchemesModel(BaseModel):
    """
    Schemes model    
    """
    project_name:str
    availables:dict

class UserModel(BaseModel):
    name:str
    
class User(BaseModel):
    username: str

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str | None = None

class ElementModel(BaseModel):
    element_id:str
    text:Optional[str] = None
    selection: Optional[str] = None
    info: Optional[str] = None
    context: Optional[dict] = None
    predict: Optional[dict] = None
    frame: Optional[list] = None

class AnnotationModel(BaseModel):
    """
    Specific Annotatoin
    """
    project_name:str
    element_id:str
    tag:str
    user:str
    scheme:str
    selection: Optional[str] = None

class SchemeModel(BaseModel):
    """
    Specific scheme
    """
    project_name:str
    name:str
    tags: Optional[list] = []

class RegexModel(BaseModel):
    """
    Regex
    """
    project_name:str
    name:str
    value:str
    user:str

class Error(BaseModel):
    error:str

class SimpleModelModel(BaseModel):
    features:list
    model:str
    params:dict|None
    scheme:str
    user:str
    standardize: Optional[bool] = True

class BertModelModel(BaseModel):
    project_name:str
    user:str
    scheme:str
    name:str
    base_model:str
    params:dict
    test_size:float

class TableElementsModel(BaseModel):
    list_ids:list
    list_labels:list
    scheme:str

class ProjectionModel(BaseModel):
    method:str
    features:list
    params:dict

class ParamsModel(BaseModel):
    params:dict