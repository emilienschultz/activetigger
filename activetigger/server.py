import os
import time
import uuid
import yaml
import concurrent.futures
from pathlib import Path
import sqlite3
import re
import json
import shutil
import pandas as pd
from pandas import DataFrame, Series
from fastapi import UploadFile
from fastapi.encoders import jsonable_encoder
from datetime import datetime, timedelta, timezone
from jose import jwt
import activetigger.functions as functions
from activetigger.models import BertModels, SimpleModels
from activetigger.datamodels import ProjectModel, SchemeModel, SimpleModelModel, UserInDB
from pydantic import ValidationError
import logging
import openai
from typing import Callable
from multiprocessing import Manager

logger = logging.getLogger('server')

class Queue():
    """
    Managining parallel processes
    For the moment : jobs in  concurrent.futures.ProcessPoolExecutor
    Comments:
        In the future, other solution ?
    """
    def __init__(self, nb_workers: int = 2):
        """
        Initiating the executor
        """
        self.nb_workers = nb_workers
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_workers)
        self.manager = Manager()
        self.current = {} # stack
        logger.info('Init Queue')

    def close(self):
        """
        Close the executor
        """
        self.executor.shutdown(cancel_futures=True, wait = False)
        self.manager.shutdown()
        logger.info('Close Queue')
        print("Executor closed, current processes:", self.state())

    def check(self):
        """
        Check if the exector still works, if not recreate it
        """
        if self.executor._broken:
            self.executor.recreate_executor()
            logger.error('Restart executor')
            print("Problem with executor ; restart")

    def recreate_executor(self):
        """
        Recreate executor
        """
        del self.executor
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_workers)

    def add(self, kind:str, func: Callable, args:dict) -> str:
        """
        Add new element to queue
        """
        # generate a unique id
        unique_id = str(uuid.uuid4()) 
        # create an event to control the function
        event = self.manager.Event()
        args["event"] = event
        # send the process
        future = self.executor.submit(func, **args)
        # save in the stack
        self.current[unique_id] = {
                                   "kind":kind, 
                                   "future":future,
                                   "event":event
                                   }
        return unique_id
    
    def kill(self, unique_id:str):
        """
        Send a kill process
        """
        if not unique_id in self.current:
            return {"error":"Id does not exist"}
        self.current[unique_id]["event"].set()
        self.delete(unique_id)
        return {"success":"Process killed"}

    def delete(self, ids:str|list):
        """
        Delete completed element ou multiple elements from the queue
        """
        if type(ids) is str:
            ids = [ids]
        for i in ids:
            if not self.current[i]["future"].done():
                print("Deleting a unfinished process")
            del self.current[i]
    
    def state(self) -> dict:
        """
        Return state of the queue
        List the stack and give the status/exception
        """
        r = {}
        for f in self.current:
            if self.current[f]["future"].running():
                info = "running"
                exception = None
            else:
                info = "done"
                exception = self.current[f]["future"].exception()
            r[f] = {"state":info, "exception":exception}
        return r

class Server():
    """
    Server to manage backend
    """
    # declare files name
    db_name:str =  "activetigger.db"
    features_file:str = "features.parquet"
    labels_file:str = "labels.parquet"
    data_file:str = "data.parquet"
    test_file:str = "test.parquet"
    default_user:str = "root"
    ALGORITHM = "HS256"
    n_workers = 2
    starting_time = None

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
        if config["path_models"] is not None:
            self.path_models = Path(config["path_models"])
        else:
            self.path_models = None
            print("Fasttext path model not specified")

        # create the database
        self.db = self.path / self.db_name
        if not self.db.exists():
            self.create_db()

        # activity of the server
        self.projects: dict = {}
        self.queue = Queue(self.n_workers)
        self.users = Users(self.db)

        # starting time
        self.starting_time = time.time()
                
    def __del__(self): 
        """
        Close the server
        """
        print("Ending the server")
        logger.error('Disconnect server')
        self.queue.executor.shutdown()
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
                description TEXT,
                created_by TEXT
                )
        '''
        cursor.execute(create_table_sql)

        # Authorizations
        create_table_sql = '''
            CREATE TABLE IF NOT EXISTS auth (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                project TEXT,
                status TEXT,
                created_by TEXT
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

        # create root user
        #self.users.add_user(self.default_user, self.default_user, role="root")
        hash_pwd = functions.get_hash(self.default_user)
        insert_query = "INSERT INTO users (user, key, description, created_by) VALUES (?, ?, ?, ?)"
        print((self.default_user, hash_pwd, "root", "system"))
        cursor.execute(insert_query, (self.default_user, hash_pwd, "root", "system"))
        conn.commit()
        conn.close()

        logger.error('Create database')

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
        logger.info(f"{action} from {user} in project {project}")

    def get_logs(self, username:str, project_name:str, limit:int):
        """
        Get logs for a user/project

        TODO : timezone for the timestamp
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        if project_name == "all":
            query = """SELECT * FROM logs WHERE user = ? ORDER BY time DESC"""
            cursor.execute(query, (username, ))
        else:
            query = """SELECT * FROM logs WHERE user = ? AND project = ? ORDER BY time DESC LIMIT ?"""
            cursor.execute(query, (username, project_name,limit))
        logs = cursor.fetchall()
        conn.commit()
        conn.close()
        df = pd.DataFrame(logs, columns = ["id", "time", "user", "project", "action", "NA"])
        return df.to_json()

    def get_session_info(self, username:str):
        """
        Get information of a username session
        """
        projects = self.users.get_auth(username)
        data = {
                "projects":[i[0] for i in projects],
                "auth": ["manager","annotator"],
                }
        return data

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

        self.projects[project_name] = Project(project_name, self.db, self.queue)
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

        Comments:
        - when saved, the files followed the nomenclature of the project : text, label, etc.
        """
        # test if possible to create the project
        if self.exists(params.project_name):
            return {"error":"Project name already exist"}
        
        params.dir = self.path / params.project_name

        if params.dir.exists():
            return {"error":"This name is already used"}
        
        os.makedirs(params.dir)

        # copy total dataset as a copy (csv for the moment)
        with open(params.dir / "data_raw.csv","wb") as f:
            f.write(file.file.read())

        # TODO : maximise the aleardy tagged in the annotate dataset, and None in the test
        # if possible, annotated data in the annotation dataset
        # if possible, test data without annotation
        # if n_test = 0, no test set
        # stratified if possible by cols_test

        # Step 1 : load all data and index to str and rename
        content = pd.read_csv(params.dir / "data_raw.csv")
        if len(content) < params.n_test + params.n_train:
            return {"error":f"Not enought data for creating the train/test dataset. Current : {len(content)} ; Selected : {params.n_test + params.n_train}"}
        content = (content.rename(columns={params.col_id:"id",params.col_text:"text"}) # rename to normalize
                           .set_index("id") # set id as index
                           .dropna(subset=['text'])) # remove NA texts
        content.index = [str(i) for i in list(content.index)] # sure to be str
    
        if params.col_label:
            content.rename(columns={params.col_label:"label"}, inplace=True)
        # TODO : also rename context columns ?

        # Information of the limit of usable text
        def limit(text):
            return 1200
        content["limit"] = content["text"].apply(limit)

        # if no label column, add dummy to act as if a column without
        if not params.col_label:
            content["label"] = None

        # Step 2 : test dataset, no already labelled data, random + stratification
        rows_test = []
        params.test = False
        if params.n_test != 0:
            # only on non labelled data
            f = content["label"].isna()
            if (f.sum()) < params.n_test:
                return {"error":"Not enought data for creating the test dataset"}
            if len(params.cols_test) == 0: # if no stratification
                testset = content[f].sample(params.n_test)
            else: #if stratification, total cat, number of element per cat, sample with a lim
                df_grouped = content[f].groupby(params.cols_test, group_keys=False)
                nb_cat = len(df_grouped)
                nb_elements_cat = round(params.n_test/nb_cat)
                testset = df_grouped.apply(lambda x: x.sample(min(len(x), nb_elements_cat)))
            testset.to_parquet(params.dir / self.test_file, index=True)
            params.test = True
            rows_test = list(testset.index)
        
        # Step 3 : train dataset, remove test rows, prioritize labelled data
        content = content.drop(rows_test)
        f_notna = content["label"].notna()
        f_na = content["label"].isna()
        if f_notna.sum()>params.n_train: # case where there is more labelled data than needed
            trainset = content[f_notna].sample(params.n_train)
        else:
            n_train_random = params.n_train - f_notna.sum() # number of element to pick
            trainset = pd.concat([content[f_notna], content[f_na].sample(n_train_random)])

        trainset.to_parquet(params.dir / self.data_file, index=True)
        trainset[["text"]+params.cols_context].to_parquet(params.dir / self.labels_file, index=True)
        trainset[[]].to_parquet(params.dir / self.features_file, index=True)

        # if the case, add labels in the database
        if (not params.col_label is None) and ("label" in trainset.columns):
            print("label case")
            df = trainset["label"].dropna()
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

        # add user right on the project + root
        self.users.set_auth(params.user, params.project_name, "manager")
        self.users.set_auth("root", params.project_name, "manager")

        # save parameters 
        params.col_label = None #reverse dummy
        self.set_project_parameters(params)
        return {"success":"Project created"}
    

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
        cursor.execute(f"DELETE FROM auth WHERE project = ?", (project_name,))
        conn.commit()
        conn.close()
        return {"success":"Project deleted"}
    
class Project(Server):
    """
    Project object
    """

    def __init__(self, 
                 project_name:str,
                 path_db:Path, 
                 queue) -> None:
        """
        Load existing project
        """
        self.starting_time = time.time()
        self.name: str = project_name
        self.db = path_db
        self.queue = queue
        self.params: ProjectModel = self.load_params(project_name)
        if self.params.dir is None:
            raise ValueError("No directory exists for this project")

        # loading data
        self.content: DataFrame = pd.read_parquet(self.params.dir / self.data_file)

        # create specific management objets
        self.schemes: Schemes = Schemes(project_name, 
                                        self.params.dir / self.labels_file, 
                                        self.params.dir / self.test_file, 
                                        self.db)
        self.features: Features = Features(project_name, 
                                           self.params.dir / self.features_file, 
                                           self.db, self.queue)
        self.bertmodels: BertModels = BertModels(self.params.dir, 
                                                 self.queue)
        self.simplemodels: SimpleModels = SimpleModels(self.params.dir, 
                                                       self.queue)
        self.zeroshot = None

    def __del__(self):
        pass

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
    
    def add_testdata(self, file, col_text, col_id, n_test):
        """
        Add a test dataset
        TODO: implement
        """
        if self.schemes.test is not None:
            return {"error":"Already a test dataset"}

        if not file.filename.endswith('.csv'):
            return {"error":"Only CSV file for the moment"}

        df = pd.read_csv(file.file, dtype={col_id:str, col_text:str}, nrows=n_test)
        
        # write the dataset
        df[[col_text]].to_parquet(self.params.dir / self.test_file)
        # load the data
        self.schemes.test = df[[col_text]]
        # update parameters
        self.params.test = True

        return {"success":"test dataset added"}

    def update_simplemodel(self, simplemodel: SimpleModelModel, 
                           username:str) -> dict:
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
        # force dfm for multi_naivebayes
        if simplemodel.model == "multi_naivebayes":
            simplemodel.features = ["dfm"]
            simplemodel.standardize = False
        
        # test if the parameters have the correct format
        try:
            validation = self.simplemodels.validation[simplemodel.model]
            r = validation(**simplemodel.params)
        except ValidationError as e:
            return {"error":e.json()}
        
        df_features = self.features.get(simplemodel.features)
        df_scheme = self.schemes.get_scheme_data(scheme=simplemodel.scheme)
        col_features = list(df_features.columns)
        data = pd.concat([df_scheme,
                          df_features],
                          axis=1)
        self.simplemodels.add_simplemodel(user = username, 
                                          scheme = simplemodel.scheme, 
                                          features=simplemodel.features, 
                                          name = simplemodel.model, 
                                          df = data, 
                                          col_labels = "labels", 
                                          col_features = col_features,
                                          model_params = simplemodel.params,
                                          standardize = simplemodel.standardize
                                          ) 
        
        return {"success":"Simplemodel updated"}

    def get_next(self,
                 scheme:str,
                 selection:str = "deterministic",
                 sample:str = "untagged",
                 user:str = "user",
                 tag:None|str = None,
                 history:list = [],
                 frame:None|list = None) -> dict:
        """
        Get next item for a specific scheme with a specific method
        - deterministic
        - random
        - active
        - maxprob
        - test
        """
        print("history",history)
        # specific case of test, random element
        if selection == "test": 
            df = self.schemes.get_scheme_data(scheme, complete=True, kind=["test"])
            f = df["labels"].isnull()
            element_id = df[f].sample(random_state=42).index[0]
            element =  {
            "element_id":str(element_id),
            "text":df.loc[element_id, "text"],
            "selection":"test",
            "context":dict(df.loc[element_id, self.params.cols_context]),
            "info":"",
            "predict":{
                    "label":None,
                    "proba":None
                    },
            "frame":[],
            "limit":1200
            }
            print(element)
            return element

        # select the current state of annotation
        df = self.schemes.get_scheme_data(scheme, complete=True)

        # build filters regarding the selection mode
        f = df["labels"].apply(lambda x : True)
        if sample == "untagged":
            f = df["labels"].isnull()
        if sample == "tagged":
            f = df["labels"].notnull()

        # manage frame selection (if projection, only in the box)
        try:
            if user in self.features.projections:
                if "data" in self.features.projections[user]:
                    projection = self.features.projections[user]["data"]
                    f_frame = (projection[0] > frame[0]) & (projection[0] < frame[2]) & (projection[1] > frame[1]) & (projection[1] < frame[3])
                    f = f & f_frame
        except:
            print("Problem on frame")

        # test if there is at least one element available
        if sum(f) == 0:
            return {"error":"No element available"}

        # select type of selection
        if selection == "deterministic": # next row
            element_id = df[f].drop(history, errors="ignore").index[0]
            indicator = None
        if selection == "random": # random row
            element_id = df[f].drop(history, errors="ignore").sample(random_state=42).index[0]
            indicator = None
        if selection == "maxprob": # higher prob, only possible if the model has been trained
            if not self.simplemodels.exists(user,scheme):
                return {"error":"Simplemodel doesn't exist"}
            if tag is None: # default label to first
                return {"error":"Select a tag"}
            sm = self.simplemodels.get_model(user, scheme) # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            element_id = proba[f][tag].drop(history, errors="ignore").sort_values(ascending=False).index[0] # get max proba id
            indicator = f"probability: {round(proba.loc[element_id,tag],2)}"
        if selection == "active": #higher entropy, only possible if the model has been trained
            if not self.simplemodels.exists(user,scheme):
                return  {"error":"Simplemodel doesn't exist"}
            sm = self.simplemodels.get_model(user, scheme) # get model
            proba = sm.proba.reindex(f.index)
            # use the history to not send already tagged data
            element_id = proba[f]["entropy"].drop(history, errors="ignore").sort_values(ascending=False).index[0] # get max entropy id
            indicator = round(proba.loc[element_id,'entropy'],2)
            indicator = f"entropy: {indicator}"
        
        # get prediction of the id if it exists
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
            "text":self.content.fillna("NA").loc[element_id,"text"],
            "context":dict(self.content.fillna("NA").loc[element_id, self.params.cols_context]),
            "selection":selection,
            "info":indicator,
            "predict":predict,
            "frame":frame,
            "limit":int(self.content.loc[element_id, "limit"])
                }

        return element
    
    def get_element(self, 
                    element_id:str,
                    scheme:str|None = None,
                    user:str|None = None):
        """
        Get an element of the database
        TODO: better homogeneise with get_next ?
        TODO: test if element exists
        """
        if not element_id in self.content.index:
            return {"error":"Element does not exist"}
        
        # get prediction if it exists
        predict = {"label":None, "proba":None}
        if (user is not None) & (scheme is not None):
            if self.simplemodels.exists(user,scheme):
                sm = self.simplemodels.get_model(user, scheme)
                predicted_label = sm.proba.loc[element_id,"prediction"]
                predicted_proba = round(sm.proba.loc[element_id,predicted_label],2)
                predict = {"label":predicted_label, 
                        "proba":predicted_proba}
        data = { 
            "element_id":element_id,
            "text":self.content.loc[element_id,"text"],
            "context":dict(self.content.fillna("NA").loc[element_id, self.params.cols_context]),
            "selection":"request",
            "predict":predict,
            "info":"get specific",
            "frame":None,
            "limit":int(self.content.loc[element_id, "limit"])
            }
        
        return {'success':data}

    def get_params(self) -> ProjectModel:
        """
        Send parameters
        """
        return self.params
    
    def get_description(self, scheme:str|None, user:str|None):
        """
        Generate a description of a current project/scheme/user
        Return:
            JSON
        """
        r = {
            "N dataset":len(self.content)
            }
        
        if scheme is None:
            return {"error":"Scheme not defined"}
        
        # part train
        df = self.schemes.get_scheme_data(scheme, kind=["add","predict"])
        r["N annotated"] = len(df)
        r["Users"] = list(self.schemes.get_distinct_users(scheme))
        r["Annotations"] = json.loads(df["labels"].value_counts().to_json())

        # part test
        df = self.schemes.get_scheme_data(scheme, kind=["test"])
        r["N test annotated"] = len(df)

        if self.simplemodels.exists(user, scheme):
            sm = self.simplemodels.get_model(user, scheme) # get model
            r["Simplemodel 10-CV"] = sm.cv10

        return r

    def get_state(self):
        """
        Send state of the project
        """
        r = {
            "params":self.params,
            "next":{
                    "methods_min":["deterministic","random"],
                    "methods":["deterministic","random","maxprob","active"],
                    "sample":["untagged","all","tagged"],
                    },
            "schemes":{
                    "available":self.schemes.available()
                    },
            "features":{
                    "options":self.features.options,
                    "available":list(self.features.map.keys()),
                    "training":list(self.features.training.keys()),
                    "infos":self.features.get_info()
                    },
            "simplemodel":{
                    "options":self.simplemodels.available_models,
                    "available":self.simplemodels.available(),
                    "training":self.simplemodels.training(),
                    },
            "bertmodels":{
                    "options":self.bertmodels.base_models,
                    "available":self.bertmodels.trained(),
                    "training":self.bertmodels.training(),
                    "test":{},
                    "base_parameters":self.bertmodels.params_default
                    },
            "projections":{
                    "available":self.features.possible_projections
                    },
            "zeroshot":{
                        "data":self.zeroshot
                        }
            }
        return  r
    
    def add_regex(self, name: str, value: str) -> dict:
        """
        Add regex to features
        """
        if name in self.features.map:
            return {"error":"a feature already has this name"}

        pattern = re.compile(value)
        f = self.content["text"].apply(lambda x: bool(pattern.search(x)))
        self.features.add(name,f)
        return {"success":"regex added"}
    
    def export_features(self, features:list, format:str = "parquet"):
        """
        Export features data in different formats
        """
        if len(features)==0:
            return {"error":"No features selected"}

        path = self.params.dir # path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.features.get(features)

        file_name = f"extract_schemes_{self.name}.{format}"

        # create files
        if format == "csv":
            data.to_csv(path / file_name)
        if format == "parquet":
            data.to_parquet(path / file_name)

        r = { "name":file_name,
              "path":path / file_name}

        return r

    def export_data(self, scheme:str, format:str = "parquet"):
        """
        Export annotation data in different formats
        """
        path = self.params.dir# path of the data
        if not path.exists():
            raise ValueError("Problem of filesystem for project")

        data = self.schemes.get_scheme_data(scheme=scheme,
                                            complete=True)
        
        file_name = f"data_{self.name}_{scheme}.{format}"

        # Create files
        if format == "csv":
            data.to_csv(path / file_name)
        if format == "parquet":
            data.to_parquet(path / file_name)

        r = {"name":file_name,"path":path / file_name}
        return r
    
    async def compute_zeroshot(self, df, params):
        """
        Zero-shot beta version
        # TODO : chunk & control the context size
        """
        r_error = ["error"] * len(df)

        # create the chunks
        # FOR THE MOMENT, ONLY 10 elements for DEMO
        if len(df)>10:
            df = df[0:10]

        # create prompt
        list_texts = "\nTexts to annotate:\n"
        for i,t in enumerate(list(df["text"])):
            list_texts += f"{i}. {t}\n"
        prompt = params.prompt + list_texts + "\nResponse format:\n{\"annotations\": [{\"text\": \"Text 1\", \"label\": \"Label1\"}, {\"text\": \"Text 2\", \"label\": \"Label2\"}, ...]}"

        # make request to client
        #client = openai.OpenAI(api_key=params.token)
        try:
            self.zeroshot = "computing"
            client = openai.AsyncOpenAI(api_key=params.token)
            print("Make openai call")
            chat_completion = await client.chat.completions.create(
                messages=[
                    {
                    "role": "system",
                    "content": """Your are a careful assistant who annotates texts for a research project. 
                    You follow precisely the guidelines, which can be in different languages.
                    """ 
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model="gpt-3.5-turbo",
                response_format={ "type": "json_object" }
                )
            print("OpenAI call done")
        except:
            self.zeroshot = None
            return {"error":"API connexion failed. Check the token."}
        # extracting results
        try:
            r = json.loads(chat_completion.choices[0].message.content)["annotations"]
            r = [i["label"] for i in r]
        except:
            return {"error":"Format problem"}
        if len(r) == len(df):
            df["zero_shot"] = r
            self.zeroshot = df[["text","zero_shot"]].reset_index().to_json()
            return {"success":"data computed"}
        else:
            return {"error":"Problem with the number of element"}
    
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
                 db_path:Path, 
                 queue) -> None:
        """
        Initit features
        """
        self.project_name = project_name
        self.path = data_path
        self.db = db_path
        self.queue = queue
        self.informations = {}
        content, map = self.load()
        self.content: DataFrame = content
        self.map:dict = map
        self.training:dict = {}

        # managing projections
        self.projections:dict = {}
        self.possible_projections:dict = {
                            "umap":{"n_neighbors":15, "min_dist":0.1, "n_components":2, "metric":'euclidean'},
                            "tsne":{"n_components":2,  "learning_rate":'auto', "init":'random', "perplexity":3}
                            }

        # options
        self.options:dict = {"sbert":{},
                        "fasttext":{},
                        "dfm":{ 
                                    "tfidf":False,
                                    "ngrams":1,
                                    "min_term_freq":5,
                                    "max_term_freq":100,
                                    "norm":None,
                                    "log":None
                                    }
                        }


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

        #print(len(self.content),len(content))
        #print(self.content)

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

        self.content = pd.concat([self.content,
                                  content],
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
        self.content = self.content.drop(columns=col)
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
        missing = []
        for i in features:
            if i in self.map:
                cols += self.map[i]
            else:
                missing.append(i)

        if len(i)>0:
            print("Missing features:", missing)
        return self.content[cols]
    
    def update_processes(self):
        """
        Check for computing processing completed
        and clean them for the queue
        """
        # for features
        for name in self.training.copy():
            unique_id = self.training[name]
            # case the process have been canceled, clean
            if not unique_id in self.queue.current:
                del self.training[name]
                continue
            # else check its state
            if self.queue.current[unique_id]["future"].done():
                r = self.queue.current[unique_id]["future"].result()
                if "error" in r:
                    print("Error in the feature processing", unique_id)
                else:
                    df = r["success"]
                    self.add(name,df)
                    self.queue.delete(unique_id)
                    del self.training[name]
                    print("Add feature", name)

        # for projections
        training = [u for u in self.projections if "queue" in self.projections[u]]
        for u in training:
            unique_id = self.projections[u]["queue"]
            if self.queue.current[unique_id]["future"].done():
                df = self.queue.current[unique_id]["future"].result()
                self.projections[u]["data"] = df
                del self.projections[u]["queue"]
                self.queue.delete(unique_id)
        
    def get_info(self):
        """
        Informations on features + update
        Comments:
            Maybe not the best solution
            Database ? How to avoid a loop ...
        """
        # update if new elements added in features
        for f in self.map:
            if ("regex_" in f) and (not f in self.informations):
                df = self.get(f)
                self.informations[f] = int(df[df.columns[0]].sum())
        return dict(self.informations)


class Schemes():
    """
    Manage project schemes & tags

    Tables :
    - schemes
    - annotations
    """
    def __init__(self,
                 project_name: str,
                 path_content:Path, # training data
                 path_test:Path, # test data
                 db_path:Path) -> None:
        """
        Init empty
        """
        self.project_name = project_name
        self.db = db_path
        self.content = pd.read_parquet(path_content) #text + context
        self.test = None
        if path_test.exists():   
            self.test = pd.read_parquet(path_test)
            
        available = self.available()

        # create a default scheme if not available
        if len(available) == 0:
            self.add_scheme(SchemeModel(project_name = project_name, 
                                 name = "default",
                                 tags = [])
                                 )

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available()}"


    def get_scheme_data(self, 
                        scheme:str, 
                        complete:bool = False, 
                        kind:list|str = ["add"]
                        ) -> DataFrame:
        """
        Get data from a scheme : id, text, context, labels
        Join with databases : content (train) or test
        """
        if not scheme in self.available():
            raise ValueError("Scheme doesn't exist")
        
        if isinstance(kind, str):
            kind = [kind]
        
        # get all elements from the db
        # - last element for each id
        # - for a specific scheme

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        action = "(" + " OR ".join([f"action = ?" for i in kind]) +")"

        query = f"""
            SELECT element_id, tag, user, MAX(time)
            FROM annotations
            WHERE scheme = ? AND project = ? AND {action}
            GROUP BY element_id
            ORDER BY time DESC;
        """
        cursor.execute(query, (scheme, self.project_name)+tuple(kind))
        results = cursor.fetchall()
        conn.close()
        df = pd.DataFrame(results, columns =["id","labels","user","timestamp"]).set_index("id")
        df.index = [str(i) for i in df.index]
        if complete: # all the elements
            if kind == "test": 
                # case if the test, join the text data
                return self.test[["text"]].join(df)
            else:
                return self.content.join(df)
        return df
    
    def get_table(self, scheme:str,
                        min:int,
                        max:int, 
                        mode:str,
                        contains:str|None = None,
                        user:str = "all"):
        """
        Get data table
        - either recent
        - or subsample of data with contains
        
        Choice to order by index.
        """
        if not mode in ["tagged","untagged","all","recent"]:
            mode = "all"
        if not scheme in self.available():
            return {"error":"scheme not available"}
        
        # data of the scheme
        df = self.get_scheme_data(scheme, complete = True)

        # case of recent annotations (no filter possible)
        if mode == "recent": 
            list_ids = self.get_recent_tags(user, scheme, max-min)
            return df.loc[list_ids]
        
        # filter for contains
        if contains:
            f_contains = df["text"].str.contains(contains)
            df = df[f_contains]
        
        # build dataset
        if mode == "tagged": 
            df = df[df["labels"].notnull()]
        if mode == "untagged":
            df = df[df["labels"].isnull()]

        # normalize size
        if max == 0:
            max = len(df)
        if  max > len(df):
            max = len(df)

        if (min > len(df)):
            return {"error":"min value too high"}
        
        return df.sort_index().iloc[min:max].drop(columns="timestamp")

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
            self.push_tag(i, None, scheme, user, "add")
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
   
    def get(self) -> dict:
        """
        state of the schemes
        """
        r = {
            "project_name":self.project_name,
            "availables":self.available()
        }
        return r

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
                 user:str = "server",
                 action:str = "add"
                 ):
        """
        Record a tag in the database
        action : add, test, predict
        """

        # test if the action is possible
        a = self.available()
        if not scheme in a:
            return {"error":"scheme unavailable"}
        if (not tag is None) and (not tag in a[scheme]):
            return {"error":"this tag doesn't belong to this scheme"}

        # TODO : add a test also for testing
        #if (not element_id in self.content.index):
        #    return {"error":"element doesn't exist"}

        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = '''
            INSERT INTO annotations (action, user, project, element_id, scheme, tag)
            VALUES (?,?,?,?,?,?);
        '''
        print("push tag")
        print((action, user, self.project_name, element_id, scheme, tag))
        cursor.execute(query, (action, user, self.project_name, element_id, scheme, tag))
        conn.commit()
        conn.close()
        return {"success":"tag added"}
    
    def push_table(self, table, user:str, action:str = "add") -> bool:
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
                            user, 
                            action)
            if "error" in r:
                return {"error":"Something happened when recording."}
        return {"success":"table pushed"}

    def get_recent_tags(self,
                    user:str,
                    scheme:str,
                    n:int) -> list:
        """
        Get the id of the n last tags added/updated
        by a user for a scheme of a project
        """
        print("get recent tags for ", user)
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
    
class Users():
    """
    Managers users
    """

    def __init__(self, db_path:Path, 
                 file_users:str = "add_users.yaml"):
        """
        Init users references
        """
        self.db = db_path
        
        # add users if add_users.yaml exists
        if Path(file_users).exists():
            existing = self.existing_users()
            with open('add_users.yaml') as f:
                add_users = yaml.safe_load(f)
            for user,password in add_users.items():
                if not user in existing:
                    self.add_user(user, password, "manager", "system")
                else:
                    print(f"Not possible to add {user}, already exists")
            # rename the file
            os.rename('add_users.yaml', 'add_users_processed.yaml')

    def get_project_auth(self, project_name:str):
        """
        Get user auth for a project
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        query = """SELECT user, status FROM auth WHERE project = ?"""
        cursor.execute(query, (project_name, ))
        auth = cursor.fetchall()
        conn.commit()
        conn.close()
        return auth
    
    def set_auth(self, username:str, project_name:str, status:str):
        """
        Set user auth for a project
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        
        # Attempt to update the entry
        update_query = "UPDATE auth SET status = ? WHERE project = ? AND user = ?"
        cursor.execute(update_query, (status, project_name, username))
        
        if cursor.rowcount == 0:
            # If no rows were updated, insert a new entry
            insert_query = "INSERT INTO auth (project, user, status) VALUES (?, ?, ?)"
            cursor.execute(insert_query, (project_name, username, status))
        conn.commit()
        conn.close()
        return {"success":"Auth added to database"}
    
    def delete_auth(self, username:str, project_name:str):
        """
        Delete user auth
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        insert_query = "DELETE FROM auth WHERE project=? AND user = ?"
        cursor.execute(insert_query, (project_name, username))
        conn.commit()
        conn.close()
        return {"success":"Auth deleted"}  

    def get_auth(self, username:str, project_name:str = "all"):
        """
        Get user auth
        """
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        if project_name == "all":
            query = """SELECT project, status FROM auth WHERE user = ?"""
            cursor.execute(query, (username,))
        else:
            query = """SELECT status FROM auth WHERE user = ? AND project = ?"""
            cursor.execute(query, (username, project_name))
        auth = cursor.fetchall()
        conn.commit()
        conn.close()
        return auth
    
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
    
    def add_user(self, 
                 name:str, 
                 password:str, 
                 role:str = "manager", 
                 created_by:str = "NA") -> bool:
        """
        Add user to database
        Comments:
            Default, users are managers
        """
        # test if the user doesn't exist
        if name in self.existing_users():
            return {"error":"Username already exists"}
        hash_pwd = functions.get_hash(password)
        # add user
        conn = sqlite3.connect(self.db)
        cursor = conn.cursor()
        insert_query = "INSERT INTO users (user, key, description, created_by) VALUES (?, ?, ?, ?)"
        cursor.execute(insert_query, (name, hash_pwd, role, created_by))
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
        conn.commit()
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
        u = UserInDB(username = name, 
                     hashed_password = user[3],
                     status = user[4])
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
    
    def auth(self, username:str, project_name:str):
        """
        Check auth for a specific project
        """
        user_auth = self.get_auth(username, project_name)
        if len(user_auth) == 0: #not associated
            return None
        return user_auth[0][0]