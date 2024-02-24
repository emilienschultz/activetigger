from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from pandas import DataFrame

class ProjectModel(BaseModel):
    """
    Parameters of a project
    """
    project_name:str
    col_text:str
    col_id:str = "index" # by default, the index
    n_rows:int = 2000
    dir:Path|None = None
    embeddings:list = []
    n_skip:int = 0
    schemes: list = []
    langage:str = "fr"
    col_tags:str|None = None # TODO: load existing tags
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
    
class ElementModel(BaseModel):
    element_id:str
    text:str|None = None

class AnnotationModel(BaseModel):
    """
    Specific Annotatoin
    """
    project_name:str
    element_id:str
    tag:str
    scheme:str = "current"

class SchemeModel(BaseModel):
    """
    Specific scheme
    """
    project_name:str
    name:str
    tags:list = []

class RegexModel(BaseModel):
    """
    Regex
    """
    project_name:str
    name:str
    value:str

class Error(BaseModel):
    error:str

class SimpleModelModel(BaseModel):
    features:list
    model:str
    params:dict|None
    scheme:str
    user:str

class BertModelModel(BaseModel):
    project_name:str
    scheme:str
    name:str
    base_model:str
    params:dict
    test_size:float