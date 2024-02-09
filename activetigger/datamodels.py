from pydantic import BaseModel
from pathlib import Path
from enum import Enum

class Action(str, Enum):
    delete = "delete"
    add = "add"
    update = "update"

class Scheme(BaseModel):
    """
    Set of labels
    """
    labels:list[str]

class ParamsModel(BaseModel):
    """
    Parameters of a project
    """
    project_name:str
    col_text:str = "text"
    n_rows:int = 2000
    dir:Path|None = None
    col_tags:str|None = None
    embeddings:list = []
    n_skip:int|None = None
    schemes: list[Scheme] = []

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
    current:str
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
    scheme:str|None = None

class SchemeModel(BaseModel):
    """
    Specific scheme
    """
    project_name:str
    name:str
    tags:list[str]

class RegexModel(BaseModel):
    """
    Regex
    """
    project_name:str
    name:str
    value:str

class SimpleModelModel(BaseModel):
    features:list
    current:str
    params:dict

class BertModelModel(BaseModel):
    name:str