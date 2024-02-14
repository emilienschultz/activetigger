from pydantic import BaseModel
from pathlib import Path
from enum import Enum
from pandas import DataFrame

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
    embeddings:list = []
    n_skip:int = 0
    schemes: list[Scheme] = []
    langage:str = "fr"
    #col_id:str = "index" TODO: select id
    col_tags:str|None = None # TODO: load existing tags
    #cols_context:list = [] TODO: select variable to keep

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
    scheme:str = "current"

class SchemeModel(BaseModel):
    """
    Specific scheme
    """
    project_name:str
    name:str
    tags:list

class RegexModel(BaseModel):
    """
    Regex
    """
    project_name:str
    name:str
    value:str

class SimpleModelModel(BaseModel):
    features:list
    model:str
    params:dict|None
    scheme:str

class BertModelModel(BaseModel):
    name:str
    col_label:str
    model_name:str = "microsoft/Multilingual-MiniLM-L12-H384"
    params:dict = {}
    test_size:float = 0.2
    #df:DataFrame = DataFrame()
    #col_text:str|None = None
    