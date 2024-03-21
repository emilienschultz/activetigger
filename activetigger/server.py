import os
import yaml # type: ignore
from datetime import datetime
import concurrent.futures
from pathlib import Path
import sqlite3
import re
import json
import shutil
import pandas as pd # type: ignore
import pyarrow.parquet as pq # type: ignore
from pandas import DataFrame, Series
from fastapi import UploadFile # type: ignore
from fastapi.encoders import jsonable_encoder # type: ignore
from datetime import datetime, timedelta, timezone
from jose import jwt # type: ignore
import activetigger.functions as functions
from activetigger.models import BertModels, SimpleModels
from activetigger.datamodels import ProjectModel, SchemesModel, SchemeModel, SimpleModelModel, UserInDB

class Server():
    """
    Server to manage backend
    """
    # files name
    db_name:str =  "activetigger.db"
    features_file:str = "features.parquet"
    labels_file:str = "labels.parquet"
    data_file:str = "data.parquet"
    test_file:str = "test.parquet"
    default_user:str = "root"
    ALGORITHM = "HS256"

    def __init__(self) -> None:
        """
        Start the server
        """
        self.time_start:datetime = datetime.now()

        # YAML configuration file
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        self.path = Path(config["path"])
        self.SECRET_KEY = config["secret_key"] #generate it automatically ?

        # Create the database
        self.db = self.path / self.db_name
        if not self.db.exists():
            self.create_db()

        # Activity of the server
        self.projects: dict = {}
        self.processes:list = []
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=2)

    def __del__(self): 
        """
        Close the server
        """
        print("Ending the server")
        self.executor.shutdown()
        print("Server off")
            
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
                element_id TEXT,
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

        # Authorizations
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS auth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                project TEXT
                  )
        '''
        cursor.execute(create_table_sql)

        # Logs
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user TEXT,
                project TEXT,
                action TEXT,
                connect TEXT
                )
        '''
        cursor.execute(create_table_sql)
        conn.commit()
        conn.close()

        # create root user
        self.add_user(self.default_user,self.default_user)

    def log_action(self, 
                   user:str, 
                   action:str, 
                   project:str = "general",
                   connect = "not implemented") -> None:
        """
        Log action in the database
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "INSERT INTO logs (user, project, action, connect) VALUES (?, ?, ?, ?)"
        cursor.execute(query, (user, project, action, connect))
        conn.commit()
        conn.close()

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
        Test if a project exists in the database
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
    
    def existing_users(self) -> list:
        """
        Get existing users
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT user FROM users"
        cursor.execute(query)
        existing_users = cursor.fetchall()
        conn.close()
        return [i[0] for i in existing_users]
    
    def add_user(self, name:str, 
                 password:str, 
                 projects = "all") -> dict:
        """
        Add user to database
        """
        # test if the user doesn't exist
        if name in self.existing_users():
            return {"error":"Username already exists"}
        hash_pwd = functions.get_hash(password)
        # add user
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        insert_query = "INSERT INTO users (user, key, projects) VALUES (?, ?, ?)"
        cursor.execute(insert_query, (name, hash_pwd, projects))
        conn.commit()
        conn.close()
        return {"success":"User added to the database"}
    
    def delete_user(self, 
                    name:str) -> dict:
        """
        Deleting user
        """
        if not name in self.existing_users():
            return {"error":"Username does not exist"}
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "DELETE FROM users WHERE user = ?"
        cursor.execute(query, (name,))
        conn.close()
        return {"success":"User deleted"}    
    
    def get_user(self, name) -> UserInDB|dict:
        """
        Get user from database
        """
        if not name in self.existing_users():
            return {"error":"Username doesn't exist"}
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "SELECT * FROM users WHERE user = ?"
        cursor.execute(query, (name,))
        user = cursor.fetchone()
        u = UserInDB(username = name, hashed_password = user[3])
        conn.close()
        return u

    def authenticate_user(self, username: str, password: str):
        """
        User authentification
        """
        user = self.get_user(username)
        if "error" in user:
            return user
        if not functions.compare_to_hash(password, user.hashed_password):
            return {"error":"Wrong password"}
        return user
    
    def create_access_token(self, data: dict, expires_min: int  = 60):
        """
        Create access token
        """
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + timedelta(minutes=expires_min)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, 
                                 self.SECRET_KEY, 
                                 algorithm=self.ALGORITHM)
        return encoded_jwt
    
    def decode_access_token(self, token:str):
        """
        Decode access token
        """
        payload = jwt.decode(token, self.SECRET_KEY, algorithms=[self.ALGORITHM])
        return payload

    def start_project(self, project_name:str) -> dict:
        """
        Load project in server
        """        
        if not self.exists(project_name):
            return {"error":"Project does not exist"}

        self.projects[project_name] = Project(project_name, self.db)
        return {"success":"Project loaded"}

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

    def create_project(self, 
                       params:ProjectModel, 
                       file:UploadFile) -> ProjectModel|dict:
        """
        Set up a new project
        - load data and save
        - initialize parameters in the db
        - initialize files
        - add preliminary tags
        """
        # create directory for the project
        if self.exists(params.project_name):
            return {"error":"Project name already exist"}
        params.dir = self.path / params.project_name
        if params.dir.exists():
            return {"error":"This name is already used"}
        os.makedirs(params.dir)

        # copy total dataset
        with open(params.dir / "data_raw.csv","wb") as f:
            f.write(file.file.read())

        # random sample of the needed data, index as str
        n_rows = params.n_train + params.n_test
        content = pd.read_csv(params.dir / "data_raw.csv").sample(n_rows)
        content = content.set_index(params.col_id)
        content.index = [str(i) for i in list(content.index)] #type: ignore
    
        # create the empty annotated file / features file
        # Put the id column as index for the rest of the treatment
        content[0:params.n_train].to_parquet(params.dir / self.data_file, index=True)
        content[params.n_train:].to_parquet(params.dir / self.test_file, index=True)
        # only for the training set for the moment
        # keep the text and the context available
        content[0:params.n_train][[params.col_text]+params.cols_context].to_parquet(params.dir / self.labels_file, index=True)
        content[0:params.n_train][[]].to_parquet(params.dir / self.features_file, index=True)

        # if the case, add labels in the database
        if (not params.col_label is None) and (params.col_label in content.columns):
            df = content[params.col_label].dropna()
            params.default_scheme = list(df.unique())
            # add the scheme in the database
            conn = sqlite3.connect(self.db)
            cursor = conn.cursor()
            query = '''
                    INSERT INTO schemes (project, name, params) 
                    VALUES (?, ?, ?)
                    '''
            cursor.execute(query, 
                        (params.project_name, 
                         "default", 
                         json.dumps(params.default_scheme)))
            conn.commit()
            # add the labels in the database
            query = '''
            INSERT INTO annotations (action, user, project, element_id, scheme, tag)
            VALUES (?,?,?,?,?,?);
            '''
            for element_id, label in df.items():
                print(("add", 
                                       params.user, 
                                       params.project_name, 
                                       element_id, 
                                       "default", 
                                       label))
                cursor.execute(query, ("add", 
                                       params.user, 
                                       params.project_name, 
                                       element_id, 
                                       "default", 
                                       label))
                conn.commit()
            conn.close()

        # save parameters 
        self.set_project_parameters(params)
        return params
    

    def delete_project(self, project_name:str) -> dict:
        """
        Delete a project
        """

        if not self.exists(project_name):
            return {"error":"Project doesn't exist"}

        # remove files
        params = self.db_get_project(project_name)
        shutil.rmtree(params.dir)

        # clean database
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM projects WHERE project_name = ?", (project_name,))
        cursor.execute(f"DELETE FROM schemes WHERE project = ?", (project_name,))
        cursor.execute(f"DELETE FROM annotations WHERE project = ?", (project_name,))
        conn.commit()
        conn.close()

        return {"success":"Project deleted"}
    

class Project(Server):
    """
    Project object
    """

    def __init__(self, 
                 project_name:str,
                 path_db:Path) -> None:
        """
        Load existing project
        """
        self.name: str = project_name
        self.db = path_db
        self.params: ProjectModel = self.load_params(project_name)
        self.content: DataFrame = pd.read_parquet(self.params.dir / self.data_file) #type: ignore
        self.schemes: Schemes = Schemes(project_name,self.params.dir / self.labels_file, self.db) #type: ignore
        self.features: Features = Features(project_name, self.params.dir / self.features_file, self.db) #type: ignore
        self.bertmodels: BertModels = BertModels(self.params.dir)
        self.simplemodels: SimpleModels = SimpleModels(self.params.dir)
        self.lock:list = [] # prevent competition

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
    
    def update_simplemodel(self, simplemodel: SimpleModelModel) -> dict:
        """
        Update simplemodel on the base of an already existing
        simplemodel object
        """
        if simplemodel.features is None or len(simplemodel.features)==0:
            return {"error":"Empty features"}
        if not simplemodel.model in list(self.simplemodels.available_models.keys()):
            return {"error":"Model doesn't exist"}
        if not simplemodel.scheme in self.schemes.available():
            return {"error":"Scheme doesn't exist"}
        if len(self.schemes.available()[simplemodel.scheme]) < 2:
            return {"error":"2 different labels needed"}
        
        df_features = self.features.get(simplemodel.features)
        df_scheme = self.schemes.get_scheme_data(scheme=simplemodel.scheme)
        col_features = list(df_features.columns)
        data = pd.concat([df_scheme,
                          df_features],
                          axis=1)
        self.simplemodels.add_simplemodel(user = simplemodel.user, 
                                          scheme = simplemodel.scheme, 
                                          features=simplemodel.features, 
                                          name = simplemodel.model, 
                                          df = data, 
                                          col_labels = "labels", 
                                          col_features = col_features,
                                          model_params = simplemodel.params,
                                          standardize = True
                                          ) 
        
        return {"success":"Simplemodel updated"}

    def get_next(self,
                 scheme:str,
                 selection:str = "deterministic",
                 sample:str = "untagged",
                 user:str = "user",
                 tag:None|str = None,
                 frame:None|list = None) -> dict:
        """
        Get next item
        Related to a specific scheme
        TODO : add lock feature
        """

        # select the current state of annotation
        df = self.schemes.get_scheme_data(scheme, complete=True)

        # build filters regarding the selection mode
        f = df["labels"].apply(lambda x : True)
        if sample == "untagged":
            f = df["labels"].isnull()
        if sample == "tagged":
            f = df["labels"].notnull()

        # manage frame selection (if projection, only in the box)
        if user in self.features.available_projections:
            if "data" in self.features.available_projections[user]:
                projection = self.features.available_projections[user]["data"]
                f_frame = (projection[0] > frame[0]) & (projection[0] < frame[2]) & (projection[1] > frame[1]) & (projection[1] < frame[3])
                f = f & f_frame
        
        val = ""

        # test if there is at least one element available
        if sum(f) == 0:
            return {"error":"No element available"}

        # select type of selection
        if selection == "deterministic": # next row
            element_id = df[f].index[0]
        if selection == "random": # random row
            element_id = df[f].sample(random_state=42).index[0]
        if selection == "maxprob": # higher prob 
            # only possible if the model has been trained
            if not self.simplemodels.exists(user,scheme):
                return {"error":"Simplemodel doesn't exist"}
            if tag is None: # default label to first
                tag = self.schemes.available()[scheme][0]
            sm = self.simplemodels.get_model(user, scheme) # get model
            element_id = sm.proba[f][tag].sort_values(ascending=False).index[0] # get max proba id
            val = f"probability: {round(sm.proba[f][tag].sort_values(ascending=False)[0],2)}"
        if selection == "active": #higher entropy
            # only possible if the model has been trained
            if not self.simplemodels.exists(user,scheme):
                return {"error":"Simplemodel doesn't exist"}
            sm = self.simplemodels.get_model(user, scheme) # get model
            element_id = sm.proba[f]["entropy"].sort_values(ascending=False).index[0] # get max entropy id
            val = round(sm.proba[f]['entropy'].sort_values(ascending=False)[0],2)
            val = f"entropy: {val}"

        # get prediction if it exists
        predict = {"label":None,
                   "proba":None}
        if self.simplemodels.exists(user,scheme):
            sm = self.simplemodels.get_model(user, scheme)
            predicted_label = sm.proba.loc[element_id,"prediction"]
            predicted_proba = round(sm.proba.loc[element_id,predicted_label],2)
            predict = {"label":predicted_label, 
                       "proba":predicted_proba}

        element =  {
            "element_id":element_id,
            "text":self.content.fillna("NA").loc[element_id,self.params.col_text],
            "context":dict(self.content.fillna("NA").loc[element_id, self.params.cols_context]),
            "selection":selection,
            "info":str(val),
            "predict":predict,
            "frame":frame
                }
        
        return element
    
    def get_element(self,element_id):
        """
        Get an element of the database
        TODO: clarify
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
    
    def get_description(self, scheme:str|None, user:str|None):
        """
        Generate a description of a project/scheme
        """
        r = {
            "N dataset":len(self.content)
            }
        
        if scheme is None:
            return None
        
        df = self.schemes.get_scheme_data(scheme)
        r["N annotated"] = len(df)
        r["Users"] = list(self.schemes.get_distinct_users(scheme))
        r["Annotations"] = json.loads(df["labels"].value_counts().to_json())

        if self.simplemodels.exists(user, scheme):
            sm = self.simplemodels.get_model(user, scheme) # get model
            r["Simplemodel 10-CV"] = sm.cv10

        return r

    def get_state(self):
        """
        Send state of the project
        """
        # update if needed
        self.bertmodels.update()

        options = {
                    "params":self.params,
                    "next":{
                        "methods":["deterministic","random","maxprob","active"],
                        "sample":["untagged","all","tagged"],
                        },
                    "schemes":{
                                "available":self.schemes.available()
                                },
                    "features":{
                            "available":list(self.features.map.keys()),
                            "training":self.features.training,
                            "options":["sbert","fasttext"]
                            },
                    "simplemodel":{ #change names existing/available to available/options
                                    "existing":self.simplemodels.available(),
                                    "available":self.simplemodels.available_models
                                },
                    "bertmodels":{
                                "options":self.bertmodels.base_models,
                                "available":self.bertmodels.trained(),
                                "training":self.bertmodels.training(),#        
                                "base_parameters":self.bertmodels.params_default
                                },
                    "projections":{
                                "available":self.features.possible_projections
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
        print("compile feature", f.shape)
        self.features.add(name,f)
        return {"success":"regex added"}
    
    def export_features(self, features:list, format:str|None = None):
        """
        Export features data in different formats
        """
        if format is None:
            format = "csv"

        path = self.path / self.name # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features)

        file_name = f"extract_schemes_{self.name}.{format}"

        if format == "csv":
            data.to_csv(path / file_name)

        if format == "parquet":
            data.to_parquet(path / file_name)

        return file_name, path / file_name


    def export_data(self, scheme:str, format:str|None = None):
        """
        Export annotation data in different formats
        """
        if format is None:
            format = "csv"

        path = self.path / self.name # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.schemes.get_scheme_data(scheme=scheme,
                                     complete=True)
        
        file_name = f"data_{self.name}_{scheme}.{format}"

        if format == "csv":
            data.to_csv(path / file_name)

        if format == "parquet":
            data.to_parquet(path / file_name)

        return file_name, path / file_name

class Features():
    """
    Manage project features
    Comment : 
    - as a file
    - use "__" as separator
    """

    def __init__(self, 
                 project_name:str, 
                 data_path:Path,
                 db_path:Path) -> None:
        """
        Initit features
        """
        self.project_name = project_name
        self.path = data_path
        self.db = db_path
        content, map = self.load()
        self.content: DataFrame = content
        self.map:dict = map
        self.training:list = []

        # managing projections
        self.possible_projections:dict = {
                            "umap":{"n_neighbors":15, "min_dist":0.1, "n_components":2, "metric":'euclidean'},
                            "tsne":{"n_components":2,  "learning_rate":'auto', "init":'random', "perplexity":3}
                            }
        self.available_projections:dict = {}


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
        var = set([i.split("__")[0] for i in data.columns])
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
        content.columns = [f"{name}__{i}" for i in content.columns]
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
        TODO : test if the feature exists
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
        

class Schemes():
    """
    Manage project schemes & tags

    Tables :
    - schemes
    - annotations
    """
    def __init__(self,
                 project_name: str,
                 path:Path, 
                 db_path:Path) -> None:
        """
        Init empty
        """
        self.project_name = project_name
        self.path = path
        self.db = db_path
        self.content = pd.read_parquet(self.path) #text + context
        available = self.available()

        # create a default scheme
        if len(available) == 0:
            self.add_scheme(SchemeModel(project_name = project_name, 
                                 name = "default",
                                 tags = [])
                                 )

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available()}"

    def get_scheme_data(self, scheme:str, complete = False) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        """
        if not scheme in self.available():
            raise ValueError("Scheme doesn't exist")
        
        # get all elements from the db
        # - last element for each id
        # - for a specific scheme

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = '''
            SELECT element_id, tag, MAX(time) AS last_timestamp
            FROM annotations
            WHERE scheme = ? AND project = ? AND action = ?
            GROUP BY element_id;
        '''
        cursor.execute(query, (scheme, self.project_name, "add"))
        results = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(results, columns =["id","labels","timestamp"]).set_index("id")
        df.index = [str(i) for i in df.index]
        if complete: # all the elements
            return self.content.join(df)
        return df
    
    def get_table(self, scheme:str,
                        min:int,
                        max:int, 
                        mode:str,
                        user:str = "all"):
        """
        Get data table
        """
        if not mode in ["tagged","untagged","all","recent"]:
            mode = "all"
        if not scheme in self.available():
            return {"error":"scheme not available"}

        # data of the scheme
        df = self.get_scheme_data(scheme, complete = True)

        # case of recent annotations
        if mode == "recent": 
            list_ids = self.get_recent_tags(user, scheme, max-min)
            return df.loc[list_ids]

        # build dataset
        if mode == "tagged": 
            df = df[df["labels"].notnull()]
        if mode == "untagged":
            df = df[df["labels"].isnull()]

        if max == 0:
            max = len(df)
        if  max > len(df):
            max = len(df)

        if (min > len(df)):
            return {"error":"min value too high"}
        
        return df.iloc[min:max].drop(columns="timestamp")

    def add_scheme(self, scheme:SchemeModel):
        """
        Add new scheme
        """
        if self.exists(scheme.name):
            return {"error":"scheme name already exists"}
        
        # add it if in database
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = '''
                INSERT INTO schemes (project, name, params) 
                VALUES (?, ?, ?)
                '''
        cursor.execute(query, 
                       (self.project_name, scheme.name, json.dumps(scheme.tags)))
        conn.commit()
        conn.close()
        return {"success":"scheme created"}

    def add_label(self, label:str, scheme:str, user:str):
        """
        Add label in a scheme
        """
        available = self.available()
        if not scheme in available:
            return {"error":"scheme doesn't exist"}
        if label in available[scheme]:
            return {"error":"label already exist"}
        labels = available[scheme]
        labels.append(label)
        print(labels)
        self.update_scheme(scheme, labels, user)
        return {"success":"scheme updated with a new label"}
    
    def delete_label(self, label:str, scheme:str, user:str):
        """
        Delete a label in a scheme
        """
        available = self.available()
        if not scheme in available:
            return {"error":"scheme doesn't exist"}
        if not label in available[scheme]:
            return {"error":"label does not exist"}
        labels = available[scheme]
        labels.remove(label)
        # push empty entry for tagged elements
        df = self.get_scheme_data(scheme)
        elements = list(df[df["labels"] == label].index)
        for i in elements:
            print(i)
            self.push_tag(i, None, scheme, user)
        self.update_scheme(scheme, labels, user)
        return {"success":"scheme updated removing a label"}

    def update_scheme(self, scheme:str, labels:list, user:str):
        """
        Update existing schemes from database
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = "UPDATE schemes SET params = ?, time_modified = ? WHERE project = ? AND name = ?"
        cursor.execute(query, (json.dumps(labels),
                               datetime.now(), 
                               self.project_name, 
                               scheme))
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
        return {"success":"scheme deleted"}
    
    def exists(self, name:str) -> bool:
        """
        Test if scheme exist
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
                   scheme:str,
                   user:str = "server") -> bool:
        """
        Delete a recorded tag
        i.e. : add empty label
        """

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = '''
            INSERT INTO annotations (action, user, project, element_id, scheme, tag)
            VALUES (?,?,?,?,?,?);
        '''
        # add delete action and then add void action
        cursor.execute(query, ("delete",user, self.project_name, element_id, scheme, None))
        cursor.execute(query, ("add",user, self.project_name, element_id, scheme, None))
        conn.commit()
        conn.close()
        return True

    def push_tag(self,
                 element_id:str, 
                 tag:str|None,
                 scheme:str,
                 user:str = "server"):
        """
        Record a tag
        """
        # test if the action is possible
        a = self.available()
        if not scheme in a:
            return {"error":"scheme unavailable"}
        if (not tag is None) and (not tag in a[scheme]):
            return {"error":"this tag doesn't belong to this scheme"}
        if not element_id in self.content.index:
            return {"error":"element doesn't exist"}
    
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = '''
            INSERT INTO annotations (action, user, project, element_id, scheme, tag)
            VALUES (?,?,?,?,?,?);
        '''
        cursor.execute(query, ("add", user, self.project_name, element_id, scheme, tag))
        conn.commit()
        conn.close()
        return {"success":"tag added"}
    
    def push_table(self, table, user:str):
        """
        Push table index/tags to update
        Comments:
        - only update modified labels
        """        
        data = {i:j for i,j in zip(table.list_ids,table.list_labels)}
        for i in data:
            r = self.push_tag(i, 
                            data[i],
                            table.scheme,
                            user)
        return {"success":"labels modified"}

    def get_recent_tags(self,
                    user:str,
                    scheme:str,
                    n:int) -> list:
        """
        Get the id of the n last tags added/updated
        by a user for a scheme of a project
        """
        print("get recent tags for ",user)
        # add case for all users

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        if user == "all": # all users
            query = """
                    SELECT DISTINCT element_id 
                    FROM annotations
                    WHERE project = ? AND scheme = ? AND action = ?
                    ORDER BY time DESC
                    LIMIT ?
                    """
            cursor.execute(query, (self.project_name,scheme, "add", n))
        else: # only one user
            query = """
                    SELECT DISTINCT element_id 
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