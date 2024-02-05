import os
import json
import sqlite3
from project import Project
from pathlib import Path
from pydantic import BaseModel
from typing import Annotated
from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File
from fastapi.encoders import jsonable_encoder
import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024 #TODO
DB_NAME = 'activetigger.db'

app = FastAPI()

# à terme toute demande de l'app aura les informations de la connexion
# nom de l'utilisateur
# accès autorisé
# projet connecté

"""
Les questions que je me pose pour le moment :
- manière de passer l'information sur le projet/utilisateur

Remarque sur fastapi
- beaucoup de validations possible pour tous les champs

Passer l'utilisateur en cookie lors de la connexion ? 
https://www.tutorialspoint.com/fastapi/fastapi_cookie_parameters.htm

Bien définir les types d'entrées / les sorties

# unwrapping pydandic object  UserInDB(**user_dict)
# untiliser l'héritage de class & le filtrage

"""

# ---------
# DataModel
# ---------

class ProjectModel(BaseModel):
    """
    Parameters of a project
    """
    project_name:str
    n_rows:int|None = None
    file:bytes|None = None
    dir:Path|None = None
    col_text:str|None = None
    col_tags:str|None = None

class UserModel(BaseModel):
    name:str
    
class ElementModel(BaseModel):
    id:str
    text:str|None = None

class Server():
    """
    Projects manager
    """
    def __init__(self, 
                 path:Path = Path("../projects")) -> None:
        """
        Start the server
        """
        logging.info("Starting server")
        logging.warning('Still under development')

        self.path:Path = path
        self.projects: dict = {}
        self.db = self.path / DB_NAME

        if not self.db.exists():
            logging.info("Creating database")
            self.create_db()
        

    def create_db(self):
        """
        Initialize the database
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS projects (
                project_name TEXT PRIMARY KEY,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT,
                time_modified TIMESTAMP
            )
        '''

        # TODO : table des utilisateurs à créer ici

        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()
        return conn

    def update_project_parameters(self, project):
        """
        Update project parameters in the DB
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT * FROM projects WHERE project_name = ?"
        cursor.execute(query, (project.project_name,))
        existing_project = cursor.fetchone()

        if existing_project:
            # Update the existing project
            update_query = "UPDATE projects SET parameters = ?, time_modified = CURRENT_TIMESTAMP WHERE project_name = ?"
            cursor.execute(update_query, (json.dumps(jsonable_encoder(project)), project.project_name))
        else:
            # Insert a new project
            insert_query = "INSERT INTO projects (project_name, parameters, time_modified) VALUES (?, ?, CURRENT_TIMESTAMP)"
            cursor.execute(insert_query, (project.project_name, json.dumps(jsonable_encoder(project))))
        conn.commit()
        conn.close()
        return True

    def db_get_project(self, project_name:str) -> ProjectModel|None:
        """
        Get project from database
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT * FROM projects WHERE project_name = ?"
        cursor.execute(query, (project_name,))
        existing_project = cursor.fetchone()
        conn.commit()
        conn.close()

        if existing_project:
            print(existing_project[2])
            p = ProjectModel(**json.loads(existing_project[2]))
            return p
        else:
            return None

    def exists(self, project_name):
        existing = self.get_existing_projects()
        v = (project_name in existing)
        return v

    def get_existing_projects(self):
        """
        Projects already existing
        """
        return [i for i in os.listdir(self.path) if os.path.isdir(self.path / i)]

    def start_project(self,project_name, **kwargs):
        """
        Initialize a project with a project_name
        """
        logging.info(f"Load project {project_name}")
        self.projects[project_name] = Project(project_name, **kwargs)

    def create_project(self, project:ProjectModel, file:UploadFile):
        """
        Create a new project
        """
        # create directory
        project.dir = self.path / project.project_name
        os.makedirs(project.dir)

        # write data
        with open(project.dir / "data.csv","wb") as f:
            f.write(file.file.read())

        # parameters of projects
        if project.col_text is None:
            project.col_text = "text"
        if not project.col_tags is None: # if existing tag column
            print("Not implemented")
        # TODO : Question of embeddings

        # save parameters
        self.update_project_parameters(project)

        return project

server = Server()

# La question est : est-ce que qu'il faut que l'utilisateur soit toujours
# mentionné ou est il possible de s'appuyer sur les cookies ?
# Dans le dependance, gérer les cookies / expiration aussi ?
    
# ------------
# Dependencies
# ------------

async def get_project(project: str) -> ProjectModel:
    """
    Fetch existing project associated with the request
    """     
    if project in server.projects:
        return server.projects[project]
    else:
        try:
            # await ??
            server.start_project(project)
            return server.projects[project]
        except KeyError:
            raise HTTPException(status_code=404, detail="Project not found")

# TODO : gérer l'authentification de l'utilisateur
async def verified_user(x_token: Annotated[str, Header()]):
    # Cookie ou header ?
    if False:
        raise HTTPException(status_code=400, detail="Invalid user")

# ------
# Routes
# ------

@app.post("/test")
async def test(project:ProjectModel,
               file: UploadFile|None = File(max_length=MAX_FILE_SIZE_BYTES)):
    return project

@app.post("/newproject", dependencies=[Depends(verified_user)])
async def new_project(project_name:str,
                      file: UploadFile,
                      n_rows:int|None = None,
                      col_text:str = "text") -> ProjectModel:

    """
    Load new project
    Parameters:
        file (file)
        n_rows (int)
    """

    # For the moment, only csv
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, 
                      detail="Only CSV file for the moment")
        
    # Test if project exist
    if server.exists(project_name):
        raise HTTPException(status_code=422, 
                detail="Project already exist")

    # Create the project
    project = ProjectModel(project_name=project_name,
                    n_rows = n_rows,
                    col_text = col_text)
    
    project = server.create_project(project, file)
    # charger un fichier + différents paramètres de création
    return project

@app.get("/e/{id}", dependencies=[Depends(verified_user)])
async def get_element(id:str, 
                      project: Annotated[Project, Depends(get_project)],
                      response_model=ElementModel) -> ElementModel:
    """
    Get specific element
    """
    try:
        e = ElementModel(**project.get_element(id))
        return e
    except: # gérer la bonne erreur
        raise HTTPException(status_code=404, detail="Element not found")

@app.get("/next")
async def get_next(mode:str|None = None, 
               on:str|None = None, 
               scheme:str|None = None): # -> ElementModel:
    return {"Get":"Next"}


















class Server_old():
    """
    Backend
    """
    def __init__(self) -> None:
        """
        Start the server
        """
        self.projects: dict = {}
        logging.warning('Still under development')
        logging.info("Start server")

    def start_project(self,project_name, **kwargs):
        """
        Initialize a project with a project_name
        """
        self.projects[project_name] = Project(project_name, **kwargs)

    # ENDPOINTS (in future FASTAPI)

    def check_credentials(self, req: dict) -> Project|dict:
        """
        Check credentials and access to project
        #TODO: users
        """
        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        p = self.projects[req["project_name"]]
        return p

    def get(self, req) -> dict:
        """
        Get data from server
        """
        logging.info(f"Get request : {req}")
        p:Project|dict = self.check_credentials(req)
        if type(p) is dict:
            return p

        if req["type"] == "next" :
            req = p.get_next(mode = req["content"]["mode"]["mode"], 
                             on = req["content"]["mode"]["on"],
                             scheme = req["content"]["scheme"]["current"])
            return req
        
        if req["type"] == "state" :
            req = p.get_state()
            return req

        if req["type"] == "element" :
            req = p.get_element(req["content"]["element_id"])
            return req
        
        if req["type"] == "schemes":
            return {
                    "type":"schemes",
                    "content":p.schemes.dump()
                    }
        
        if req["type"] == "simplemodel":
            return {
                "type":"simplemodel",
                "content":p.simplemodel.get_params()
            }
        
        if req["type"] == "bert":
            return {
                "type":"bert",
                "content":p.bertmodel.get_params()
            }
                
        return {"error":"request not found"}
    
    def post(self, req:dict) -> dict:
        """
        Manage post requests
        """
        logging.info(f"Post request : {req}")
        p:Project|dict = self.check_credentials(req)
        if type(p) is dict:
            return p

        if req["type"] == "label" :
            p.add_label(req["content"]["element_id"],
                        req["content"]["label"])
            return {"add_label":"success"}
        
        if req["type"] == "delete_label":
            p.delete_label(req["content"]["element_id"])
            return {"delete_label":"success"}            

        if req["type"] == "update_schemes":
            p.update_schemes(req["content"])
            return {"update_schemes":"success"}
        
        if req["type"] == "simplemodel":
            # train a new simple model
            return p.update_simplemodel(req["content"])
        
        if req["type"] == "regex":
            return p.add_regex(req["content"]["name"],req["content"]["value"])
        
        if req["type"] == "delete_feature":
            return p.features.delete(req["content"]["name"])
        
        if req["type"] == "new_scheme":
            if p.schemes.add(req["content"]["name"],[]):
                return {"new_scheme":"created"}
            else:
                return {"error":"new scheme not created"}
        
        if req["type"] == "train_bert":
            return p.bertmodel.start_training(req["content"])

        return {"error":"request not found"}
