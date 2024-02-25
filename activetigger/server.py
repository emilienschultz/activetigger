import os
from datetime import datetime
import concurrent.futures
from pathlib import Path
import sqlite3
import pandas as pd # type: ignore
import re
import pyarrow.parquet as pq # type: ignore
import json
import functions
from models import BertModels, SimpleModels
from datamodels import ProjectModel, SchemesModel, SchemeModel, SimpleModelModel
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
    default_user:str = "user"
    n_workers = 4 #os.cpu_count()


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
        self.time_start:datetime = datetime.now()
        self.processes:list = []
        # to deal multiprocessing TODO: move to Processes ?
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers=self.n_workers)

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
            p = ProjectModel(**json.loads(existing_project[2]))
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

    def set_project_parameters(self, project: ProjectModel) -> dict:
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
                       params:ProjectModel, 
                       file:UploadFile) -> ProjectModel:
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
        self.params: ProjectModel = self.load_params(project_name)
        self.content: DataFrame = pd.read_parquet(self.params.dir / self.data_file) #type: ignore
        self.schemes: Schemes = Schemes(project_name, 
                                        self.params.dir / self.labels_file) #type: ignore
        self.features: Features = Features(project_name,
                                           self.params.dir / self.features_file) #type: ignore
        self.bertmodels: BertModels = BertModels(self.params.dir)
        self.simplemodels: SimpleModels = SimpleModels()

        # Compute features if requested
        if ("sbert" in self.params.embeddings) & (not "sbert" in self.features.map):
            self.compute_embeddings(emb="sbert")
        if ("fasttext" in self.params.embeddings) & (not "fasttext" in self.features.map):
            self.compute_embeddings(emb="fasttext")

    def load_params(self, project_name:str) -> ProjectModel:
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
            return ProjectModel(**json.loads(existing_project[2]))
        else:
            raise NameError(f"{project_name} does not exist.")

    async def compute_embeddings(self,
                           emb:str) -> dict:
        """
        Compute embeddings
        TODO : AS A SUBPROCESS
        """
        print("start compute embeddings ",emb, self.content.shape)
        if emb == "fasttext":
            f = functions.to_fasttext
            #emb_fasttext = functions.to_fasttext(self.content[self.params.col_text])
            #self.features.add("fasttext", emb_fasttext)
            #return {"success":"fasttext embeddings computed"}
        if emb == "sbert":
            f = functions.to_sbert
            #emb_sbert = functions.to_sbert(self.content[self.params.col_text])
            #self.features.add("sbert", emb_sbert)
            #return {"success":"sbert embeddings computed"}
        #future = self.pool.submit(f, self.content[self.params.col_text])
        #def call_back(future):
        #    df = future.results()
        #future.add_done_callback(call_back)
        calc_emb = f(self.content[self.params.col_text])
        self.features.add(emb, calc_emb)
        return {"success":f"{emb} embeddings computed"}

        raise NameError(f"{emb} does not exist.")
    
    def fit_simplemodel(self,
                        model:str,
                        features:list|str,
                        scheme:str,
                        user:str = "user",
                        model_params: None|dict = None
                        ) -> bool:
        """
        Create and fit a simple model with project data
        """

        # build the dataset with label + predictors
        df_features = self.features.get(features)
        col_labels = self.schemes.col_name(scheme)
        col_features = list(df_features.columns)
        data = pd.concat([self.schemes.content[col_labels],
                          df_features],
                          axis=1)
        self.simplemodels.add_simplemodel(user = user, 
                                          scheme = scheme, 
                                          features=features, 
                                          name = model, 
                                          df = data, 
                                          col_labels = col_labels, 
                                          col_features = col_features,
                                          model_params = model_params,
                                          standardize = True
                                          ) 
        
        #s = SimpleModel(model=model,
        #                data = data,
        #                col_label = col_label,
        #                col_predictors = col_predictors,
        #                model_params=model_params
        #                )
        return True
    
    def update_simplemodel(self, simplemodel: SimpleModelModel) -> dict:
        if simplemodel.features is None or len(simplemodel.features)==0:
            return {"error":"Empty features"}
        if not simplemodel.model in list(self.simplemodels.available_models.keys()):
            return {"error":"Model doesn't exist"}
        if not simplemodel.scheme in self.schemes.available():
            return {"error":"Scheme doesn't exist"}
        self.fit_simplemodel(
                            model=simplemodel.model,
                            features=simplemodel.features,
                            scheme=simplemodel.scheme,
                            user =simplemodel.user,
                            model_params=simplemodel.params
                            )
        return {"success":"new simplemodel"}
    
    def get_next(self,
                 scheme:str,
                 selection:str = "deterministic",
                 sample:str = "untagged",
                 user:str = "user",
                 tag:None|str = None) -> dict:
        """
        Get next item
        Related to a specific scheme
        """

        # Select the sample
        f = self.schemes.content[scheme].apply(lambda x : True)
        if sample == "untagged":
            f = self.schemes.content[scheme].isnull()
        if sample == "tagged":
            f = self.schemes.content[scheme].notnull()

        # Type of selection
        if selection == "deterministic": # next row
            element_id = self.schemes.content[f].index[0]
        if selection == "random": # random row
            element_id = self.schemes.content[f].sample(random_state=42).index[0]
        if selection == "maxprob": # higher prob 
            # only possible if the model has been trained
            if not self.simplemodels.exists(user,scheme):
                return {"error":"Simplemodel doesn't exist"}
            if tag is None: # default label to first
                tag = self.schemes.available()[scheme][0]
            sm = self.simplemodels.get_model(user, scheme) # get model
            element_id = sm.proba[f][tag].sort_values(ascending=False).index[0] # get max proba id
        
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

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params
    
    def get_stats_annotations(self, scheme:str):

        df = self.schemes.get_scheme_data(scheme)
        df["labels"].value_counts()

        stats = {
                    "dataset total":len(self.content),
                    "annotated elements":len(df),
                    "different users":list(self.schemes.get_distinct_users(scheme)),
                    "annotations distribution":json.loads(df["labels"].value_counts().to_json()),
                    "last annotation":"TO DO",
                }
        return stats

    def get_state(self):
        """
        Send state of the project
        """
        options = {
                    "params":self.params,
                    "next":{
                        "methods":["deterministic","random","maxprob"],
                        "sample":["untagged","all","tagged"],
                        },
                    "schemes":{
                                "available":self.schemes.available()
                                },
                    "features":{
                            "available":list(self.features.map.keys())
                            },
                    "simplemodel":{ #change names existing/available to available/options
                                    "existing":self.simplemodels.available(),
                                    "available":self.simplemodels.available_models
                                    },
                    "bertmodel":{
                                "options":self.bertmodels.base_models,
                                "available":self.bertmodels.trained(),
                                "training":self.bertmodels.training()
                                }
                   }
        # TODO : change available label to default ... 
        return  options
    
    def add_regex(self, name: str, value: str) -> dict:
        """
        Add regex to features
        """
        if name in self.features.map:
            return {"error":"a feature already has this name"}

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
        self.col = None
        self.content = pd.read_parquet(self.path)

        # Initialize the current scheme
        available = self.available()
        if len(available) == 0: #if no scheme available -> default
            self.add_scheme(SchemeModel(project_name=project_name, 
                                 name = "default",
                                 tags= []))
            #self.select("default")
        #else: #else, select the first
        #    self.select(list(available.keys())[0])

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
    
    def get_table_elements(self, 
                           scheme:str,
                           min:int,
                           max:int, 
                           mode:str,
                           user:str = "user"):
        """
        Get json data table for an interval
        """
        if not mode in ["tagged","untagged","all","recent"]:
            mode = "all"

        if not scheme in self.available():
            return {"error":"scheme not available"}

        df = self.content.loc[:,[self.content.columns[0],scheme]]
        df.columns = ["text","labels"]

        if max == 0:
            max = len(df)
        if  max > len(df):
            max = len(df)

        if (min > len(df)):
            return {"error":"min value too high"}

        if mode == "recent": # get recent annotations
            print(user,scheme, max-min)
            list_ids = self.get_recent_tags(user,scheme, max-min)
            return df.loc[list_ids]

        # TODO : user ?
        if mode == "tagged": 
            df = df[df["labels"].notnull()]
        if mode == "untagged":
            df = df[df["labels"].isnull()]

        return df.iloc[min:max]

    def save_data(self) -> None:
        """
        Save the element to file
        """
        self.content.to_parquet(self.path)

    def col_name(self, s:str):
        """
        Association name - column
        (for the moment 1 - 1)
        """
        return s

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
                 scheme:str,
                 user:str = "user"):
        """
        Record a tag
        """
        # test if the action is possible
        a = self.available()
        if not scheme in a:
            return {"error":"scheme unavailable"}
        if not tag in a[scheme]:
            return {"error":"this tag doesn't belong to this scheme"}
        if not element_id in self.content.index:
            return {"error":"element doesn't exist"}
    
        # return message
        if not self.content.loc[element_id, scheme] is None:
            r = {"success":"tag updated"}
        else:
            r = {"success":"tag added"}

        # make action
        self.content.loc[element_id, scheme] = tag
        self.log_action("add", user, element_id, scheme, tag)
        self.save_data()
        return r

    def get_recent_tags(self,
                    user:str,
                    scheme:str,
                    n:int) -> list:
        """
        Get the id of the n last tags added/updated
        by a user for a scheme of a project
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = """
                SELECT element_id, time 
                FROM annotations
                WHERE project = ? AND user = ? AND scheme = ? AND action = ?
                ORDER BY time DESC
                LIMIT ?
                """
        cursor.execute(query, (self.project_name,user,scheme, "add", n))
        results = cursor.fetchall()
        conn.commit()
        conn.close()
        return [i[0] for i in results]
    
    def get_distinct_users(self, scheme:str):
        """
        Get users action for a scheme
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = """
                SELECT DISTINCT user 
                FROM annotations
                WHERE project = ? AND scheme = ? AND action = ?
                """
        cursor.execute(query, (self.project_name,scheme, "add"))
        results = cursor.fetchall()
        conn.commit()
        conn.close()
        return results

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