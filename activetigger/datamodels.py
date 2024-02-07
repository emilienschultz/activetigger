from pydantic import BaseModel
from pathlib import Path

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

class UserModel(BaseModel):
    name:str
    
class ElementModel(BaseModel):
    id:str
    text:str|None = None