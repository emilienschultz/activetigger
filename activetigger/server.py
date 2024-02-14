import os
from datetime import datetime
from pathlib import Path
import sqlite3
import pandas as pd # type: ignore
import re
import pyarrow.parquet as pq # type: ignore
import json
import functions
from models import SimpleModel, BertModel
from datamodels import ParamsModel, SchemesModel, SchemeModel, SimpleModelModel
from pandas import DataFrame, Series
from fastapi import UploadFile # type: ignore
from fastapi.encoders import jsonable_encoder # type: ignore
import shutil
import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
class Session():
    """
    Global session parameters
    TODO : load from yaml file
    """
    logging.info("Starting session")
    logging.warning('Still under development')

    max_file_size: int = 50 * 1024 * 1024
    path: Path = Path("../projects")
    db: Path = Path('../projects/activetigger.db')
    features_file:str = "features.parquet"
    labels_file:str = "labels.parquet"
    data_file:str = "data.parquet"


class Server(Session):
    """
    Server to manage projects
    """
    def __init__(self) -> None:
        """
        Start the server
        """
        logging.info("Starting server")

        self.projects: dict = {}
        self.time_start = datetime.now()

        if not self.db.exists():
            logging.info("Creating database")
            self.create_db()

    def create_db(self) -> None:
        """
        Initialize the database
        """

        # create the repertory if needed
        if not self.path.exists():
            os.makedirs(self.path)

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()

        # Projects table
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS projects (
                project_name TEXT PRIMARY KEY,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                parameters TEXT,
                time_modified TIMESTAMP,
                user TEXT
            )
        '''
        cursor.execute(create_table_sql)

        # Schemes table
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS schemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time_created TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                time_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                project TEXT,
                name TEXT,
                params TEXT
            )
        '''
        cursor.execute(create_table_sql)

        # Annotation history table
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                action TEXT,
                user TEXT,
                project TEXT,
                element_id,
                scheme TEXT,
                tag TEXT
            )
        '''
        cursor.execute(create_table_sql)

        # User table
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                key TEXT,
                projects TEXT
                  )
        '''
        cursor.execute(create_table_sql)

        conn.commit()
        conn.close()
        return None

    def db_get_project(self, project_name:str) -> ParamsModel|None:
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
            p = ParamsModel(**json.loads(existing_project[2]))
            return p
        else:
            return None

    def exists(self, project_name) -> bool:
        """
        Test if a project exists
        """
        existing = self.existing_projects()
        v = (project_name in existing)
        return v
    
    def existing_projects(self) -> list:
        """
        Get existing projects
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT project_name FROM projects"
        cursor.execute(query)
        existing_project = cursor.fetchall()
        conn.close()
        return [i[0] for i in existing_project]
    
    def start_project(self, project_name:str) -> bool:
        """
        Load project in memory
        """        
        if not self.exists(project_name):
            raise ValueError("Project don't exist")

        logging.info(f"Load project {project_name}")
        self.projects[project_name] = Project(project_name)
        return True

    def set_project_parameters(self, project: ParamsModel) -> dict:
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
        return {"success":"project updated"}
    
    def remove_project_parameters(self, project_name:str) -> bool:
        """
        Delete database entry
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM projects WHERE project_name = ?", (project_name,))
        cursor.execute(f"DELETE FROM schemes WHERE project = ?", (project_name,))
        conn.commit()
        conn.close()
        return True

    def create_project(self, 
                       params:ParamsModel, 
                       file:UploadFile) -> ParamsModel:
        """
        Set up a new project
        - load data and save
        - initialize parameters
        - initialize files
        """
        # create directory
        params.dir = self.path / params.project_name
        os.makedirs(params.dir)

        # write data
        with open(params.dir / "data_raw.csv","wb") as f:
            f.write(file.file.read())

        # save parameters 
        self.set_project_parameters(params)

        # load only the number of rows for the project
        content = pd.read_csv(params.dir / "data_raw.csv", 
                                index_col=0, 
                                nrows=params.n_rows)
        
        content.index = [str(i) for i in list(content.index)] #type: ignore
        print(content.index)
    
        # create the empty annotated file / features file
        content.to_parquet(params.dir / self.data_file, index=True)
        content[[params.col_text]].to_parquet(params.dir / self.labels_file, index=True)
        content[[]].to_parquet(params.dir / self.features_file, index=True)

        """
        TODO : deal if already tagged column
        if "col_tags" in kwargs: # if existing tag column
            self.content[self.schemes.col] = self.content[kwargs["col_tags"]]
        else:
            self.content[self.schemes.col] = None
        """

        return params

    def delete_project(self, project_name:str) -> dict:
        """
        Delete a project
        """

        if self.exists(project_name):
            params = self.db_get_project(project_name)
            self.remove_project_parameters(project_name)
            shutil.rmtree(params.dir)
            return {"success":"project deleted"}
        else:
            return {"error":"project doesn't exist"}

class Project(Session):
    """
    Project object
    """

    def __init__(self, 
                 project_name: str) -> None:
        """
        Load existing project
        """
        self.name: str = project_name
        self.params: ParamsModel = self.load_params(project_name)
        self.content: DataFrame = pd.read_parquet(self.params.dir / self.data_file) #type: ignore
        self.schemes: Schemes = Schemes(project_name, 
                                        self.params.dir / self.labels_file) #type: ignore
        self.features: Features = Features(project_name,
                                           self.params.dir / self.features_file) #type: ignore
        self.bertmodel: BertModel = BertModel(self.params.dir)
        self.simplemodel: SimpleModel = SimpleModel()

        # Compute features if requested
        if ("sbert" in self.params.embeddings) & (not "sbert" in self.features.map):
            self.compute_embeddings(emb="sbert")
        if ("fasttext" in self.params.embeddings) & (not "fasttext" in self.features.map):
            self.compute_embeddings(emb="fasttext")

    def load_params(self, project_name:str) -> ParamsModel:
        """
        Load params from database
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT * FROM projects WHERE project_name = ?"
        cursor.execute(query, (project_name,))
        existing_project = cursor.fetchone()
        conn.commit()
        conn.close()

        if existing_project:
            return ParamsModel(**json.loads(existing_project[2]))
        else:
            raise NameError(f"{project_name} does not exist.")

    def compute_embeddings(self,
                           emb:str) -> dict:
        """
        Compute embeddings
        TODO : ASYNC TO TEST HERE
        """
        print("start compute embeddings ",emb, self.content.shape)
        if emb == "fasttext":
            emb_fasttext = functions.to_fasttext(self.content[self.params.col_text])
            self.features.add("fasttext", emb_fasttext)
            return {"success":"fasttext embeddings computed"}
        if emb == "sbert":
            emb_sbert = functions.to_sbert(self.content[self.params.col_text])
            self.features.add("sbert", emb_sbert)
            return {"success":"sbert embeddings computed"}
        raise NameError(f"{emb} does not exist.")
    
    def fit_simplemodel(self,
                        model:str,
                        features:list|str,
                        scheme:str,
                        model_params: None|dict = None
                        ) -> SimpleModel:
        """
        Create and fit a simple model with project data
        """

        # build the dataset with label + predictors

        df_features = self.features.get(features)
        col_label = self.schemes.col_name(scheme)
        col_predictors = df_features.columns
        data = pd.concat([self.schemes.content[col_label],
                          df_features],
                          axis=1)
        
        s = SimpleModel(model=model,
                        data = data,
                        col_label = col_label,
                        col_predictors = col_predictors,
                        model_params=model_params
                        )
        return s
    
    def update_simplemodel(self, simplemodel: SimpleModelModel) -> dict:
        if simplemodel.features is None or len(simplemodel.features)==0:
            return {"error":"no features"}
        self.simplemodel = self.fit_simplemodel(
                                model=simplemodel.model,
                                features=simplemodel.features,
                                scheme=simplemodel.scheme,
                                model_params=simplemodel.params
                                )
        return {"success":"new simplemodel"}
    

    def get_next(self,
                 scheme:str,
                 mode:str = "deterministic",
                 on:str = "untagged",
                 tag:None|str = None) -> dict:
        """
        Get next item
        Related to a specific scheme

        TODO : gérer les cases tagguées/non tagguées etc.
        """

        # check if the current scheme is selected
        if not self.schemes.name == scheme:
            print("Change of scheme")
            self.schemes.select(scheme)

        # Pour le moment uniquement les cases non nulles
        f = self.schemes.content[self.schemes.col].isnull()

        if mode == "deterministic": # next row
            element_id = self.schemes.content[f].index[0]
        if mode == "random": # random row
            element_id = self.schemes.content[f].sample(random_state=42).index[0]
        if mode == "maxprob": # higher prob row
            if self.simplemodel.name is None: # if no model, build default
                print("Build default simple model")
                self.simplemodel = self.fit_simplemodel(model = "liblinear",
                                                        features = "all",
                                                        scheme=self.schemes.name
                                                        )
            if tag is None: # default label to first
                tag = self.schemes.labels[0] #type: ignore

            # higher predict value
            element_id = self.simplemodel.proba[f][tag].sort_values(ascending=False).index[0] #type: ignore
        
        # TODO : put a lock on the element when sent ?

        # Pour le moment uniquement l'id et le texte (dans le futur ajouter tous les éléments)
        return  self.get_element(element_id)
    
    def get_element(self,element_id):
        """
        Get an element of the database
        """
        columns = ["text"]
        return {"element_id":element_id,
                "text":self.content.loc[element_id,"text"]
                }

    def get_params(self) -> ParamsModel:
        """
        Send parameters
        """
        return self.params
    
    def get_state(self) -> dict:
        """
        Send state of the project
        """
        selection_available = ["deterministic","random"]
        if self.simplemodel.name is not None:
            selection_available.append("maxprob")

        options = {
                    "params":self.params,
                    "modes":{
                        "available_modes":selection_available,
                        # For the moment mode/on/label are not in the state project side
                        # default values
                        "mode":"deterministic",
                        "on":"untagged",
                        "label":None
                        },
                    "schemes":{
                                "current":self.schemes.name,
                                "available":self.schemes.available()
                                },
                    "features":{
                            "available_features":self.features.map.keys()
                          }
                   }
        
        return  options
    
    def add_regex(self, name: str, value: str) -> dict:
        """
        Add regex to features
        """
        pattern = re.compile(value)
        f = self.content[self.params.col_text].apply(lambda x: bool(pattern.search(x)))
        self.features.add(name,f)
        return {"success":"regex added"}
    

class Features(Session):
    """
    Manage project features
    Comment : as a file
    """

    def __init__(self, 
                 project_name:str, 
                 data_path:Path) -> None:
        """
        Initit features
        """
        self.project_name = project_name
        self.path = data_path
        content, map = self.load()
        self.content: DataFrame = content
        self.map:dict = map

    def __repr__(self) -> str:
        return f"Available features : {self.map}"
    
    def load(self):
        """
        Load file and agregate columns
        """
        def find_strings_with_pattern(strings, pattern):
            matching_strings = [s for s in strings if re.match(pattern, s)]
            return matching_strings
        data = pd.read_parquet(self.path)
        var = set([i.split("_")[0] for i in data.columns])
        dic = {i:find_strings_with_pattern(data.columns,i) for i in var}
        return data, dic
    
    def add(self, 
            name:str, 
            content:DataFrame|Series) -> dict:
        """
        Add feature(s) and save
        """

        # test length
        if len(content) != len(self.content):
            raise ValueError("Features don't have the right shape") 
        
        if name in self.map:
            return {"error":"feature name already exists"}

        # change type
        if type(content)==Series:
            content = pd.DataFrame(content)

        # add to the table & dictionnary
        content.columns = [f"{name}_{i}" for i in content.columns]
        self.map[name] = list(content.columns)
        self.content = pd.concat([self.content,content],
                                     axis=1)
        # save
        self.content.to_parquet(self.path)
        return {"success":"feature added"}

    def delete(self, name:str):
        """
        Delete feature
        """
        if not name in self.map:
            return {"error":"feature doesn't exist"}

        col = self.get([name])
        del self.map[name]
        self.content.drop(columns=col)
        self.content.to_parquet(self.path)
        return {"success":"feature deleted"}
            
    def get(self, features:list|str = "all"):
        """
        Get content for specific features
        """
        if features == "all":
            features = list(self.map.keys())
        if type(features) is str:
            features = [features]

        cols = []
        for i in features:
            if i in self.map:
                cols += self.map[i]

        return self.content[cols]

class Schemes(Session):
    """
    Manage project schemes & tags
    """
    def __init__(self,
                 project_name: str,
                 path:Path) -> None:
        """
        Init empty
        """
        self.project_name = project_name
        self.path = path
        self.name = None
        self.labels = None
        self.col = None

        # Load data
        self.content = pd.read_parquet(self.path)

        # Initialize the current scheme
        available = self.available()
        if len(available) == 0: #if no scheme available -> default
            self.add_scheme(SchemeModel(project_name=project_name, 
                                 name = "default",
                                 tags= []))
            self.select("default")
        else: #else, select the first
            self.select(list(available.keys())[0])

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available()}"

    def get_scheme_data(self, s:str) -> DataFrame:
        """
        Get dataframe of a scheme

        Comment : first column is the text
        """
        if not s in self.available():
            raise ValueError("Scheme doesn't exist")
        
        df = self.content.loc[:,[self.content.columns[0],s]].dropna()
        df.columns = ["text","labels"]
        return df

    def save_data(self) -> None:
        """
        Save the element to file
        """
        self.content.to_parquet(self.path)

    def col_name(self, s = None):
        """
        Association name - column
        (for the moment 1 - 1)
        """
        if not s:
            return self.name
        return s

    def select(self, name) -> None:
        """
        Select current scheme
        """
        available = self.available()
        if name in available:
            self.name = name
            self.labels = available[name]
            self.col = self.col_name()
        else:
            raise IndexError

    def add_scheme(self, scheme:SchemeModel):
        """
        Add new scheme
        """
        if self.exists(scheme.name):
            return {"error":"scheme name already exists"}
        
        # add it if in database
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "INSERT INTO schemes (project, name, params) VALUES (?, ?, ?)"
        cursor.execute(query, (self.project_name, scheme.name, json.dumps(scheme.tags)))
        conn.commit()
        conn.close()

        # add it in the file
        self.content[scheme.name] = None
        self.save_data()

        return {"success":"scheme created"}

    def update_scheme(self, scheme:SchemeModel):
        """
        Update existing schemes from database
        """
        if not self.exists(scheme.name):
            return {"error":"scheme doesn't exist in db"}
        
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "UPDATE schemes SET params = ?, time_modified = ? WHERE project = ? AND name = ?"
        cursor.execute(query, (json.dumps(scheme.tags),
                               datetime.now(), 
                               self.project_name, 
                               scheme.name))
        conn.commit()
        conn.close()
        return {"success":"scheme updated"}
        
    def delete_scheme(self, scheme:SchemeModel) -> dict:
        """
        Delete a scheme
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "DELETE FROM schemes WHERE project = ? AND name = ?" 
        cursor.execute(query, (self.project_name,scheme.name))
        conn.commit()
        conn.close()

        self.content.drop(columns=scheme.name)
        self.save_data()

        return {"success":"scheme deleted"}
    
    def exists(self, name:str) -> bool:
        """
        Test if scheme exist
        TODO : harmoniser avec le fichier
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT * FROM schemes WHERE project = ? AND name = ?"
        cursor.execute(query, (self.project_name,name))
        result = cursor.fetchone()
        conn.close()
        if result is None:
            return False
        else:
            return True

    def available(self) -> dict:
        """
        Available schemes
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT name, params FROM schemes WHERE project = ?"
        cursor.execute(query, (self.project_name,))
        results = cursor.fetchall()
        conn.commit()
        conn.close()
        return {i[0]:json.loads(i[1]) for i in results}
   
    def get(self) -> SchemesModel:
        """
        state of the schemes
        """
        return SchemesModel(project_name=self.project_name,
                            current = self.name, #type: ignore
                            availables=self.available()
                            )


    def delete_tag(self, 
                   element_id:str,
                   scheme:str) -> bool:
        """
        Delete a recorded tag
        """
        self.content.loc[element_id,scheme] = None
        return True

    def push_tag(self,
                 element_id:str, 
                 tag:str,
                 scheme:str = "current",
                 user:str = "user"):
        """
        Record a tag
        """
        if scheme == "current":
            scheme = self.col #type: ignore

        if not scheme in self.available():
            return {"error":"scheme unavailable"}
        
        if not element_id in self.content.index:
            return {"error":"element doesn't exist"}
        
        # TODO : test if the tag is in in the tags

        if not self.content.loc[element_id, scheme] is None:
            r = {"success":"tag updated"}
        else:
            r = {"success":"tag added"}
        self.content.loc[element_id, scheme] = tag
        self.log_action("add", user, element_id, self.name, tag)
        self.save_data()
        return r

    def log_action(self, 
                   action:str, 
                   user:str, 
                   element_id:str,
                   scheme:str, 
                   tag:str) -> bool:
        """
        Add annotation log
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "INSERT INTO annotations (action, user, project, element_id, scheme, tag) VALUES (?, ?, ?, ?, ?, ?)"
        cursor.execute(query, (action, user, self.project_name, element_id, scheme, tag))
        conn.commit()
        conn.close()
        return True