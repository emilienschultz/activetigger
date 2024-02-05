"""
Classes available : Projet, Features, Schemes 
"""
import os
from pathlib import Path
import yaml # type: ignore
import pandas as pd # type: ignore
import re
import pyarrow.parquet as pq # type: ignore
import json
import functions
from models import SimpleModel, BertModel
from pandas import DataFrame, Series
from pydantic import BaseModel

import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# optimiser le calcul des embeddings avec GPU
# lancer un processus indépendant pour calculer les embeddings
# use Pydantic & BaseModel to type data in API and automatically
# generate the JSON content https://realpython.com/api-integration-in-python/ 

# plus précisémment définir le format des objets qui circulent entre le back et le front
# - element
# - options
# - liste d'éléments

# TODO : gérer la sauvegarde de tous les éléments
# - schemes gérés avec les paramètres (mais ça doit changer)
# - features gestion à part
# - data + coding

# TODO : gérer les éléments tagged dans next etc.

class Project():
    """
    Project (database/params)
    """

    def __init__(self, project_name : str, **kwargs) -> None:
        """
        Initialize a project 
        (load or create)
        """

        self.name: str = project_name
        self.schemes: Schemes = Schemes(self.name)
        self.features = Features(project_name=self.name)
        self.params: dict = {}
        self.content: DataFrame = DataFrame()
        self.simplemodel:SimpleModel = SimpleModel()
        self.bertmodel:BertModel = BertModel(path = f"{self.name}")

        # load or create project    
        if self.exists(project_name):
             self.load_params()
             self.load_data()
             self.features.load()
        else:
             self.create(project_name, **kwargs)
    
    def exists(self, project_name: str) -> bool:
        """
        Test if a project exists
        """
        return Path(f"{project_name}/{project_name}.yaml").exists()

    def create(self, project_name:str, **kwargs):
        """
        Create new project
        """

        ## TO DO : tests que tous les paramètres ont bien été renseignés
        # Le premier schéma enregistré est appelé default
        
        if not Path(project_name).exists():
            os.makedirs(project_name)
        else:
            print("Erreur, le dossier existe") # gérer la gestion des erreurs

        # Load data
        self.content = pd.read_csv(kwargs["file"],
                                   index_col=0,
                                   low_memory=False,
                                   nrows=kwargs["n_rows"])

        # Manage parameters
        if not "col_text" in kwargs:
            kwargs["col_text"] = "text"
        if "col_tags" in kwargs: # if existing tag column
            kwargs["cat"] = list(self.content[kwargs["col_tags"]].dropna().unique())
        if not "sbert" in kwargs:
            kwargs["sbert"] = False
        if not "fasttext" in kwargs:
            kwargs["fasttext"] = False

        # TODO : test que les colonnes utilisées existent bien
            
        self.params = {
                    "project_name":project_name,
                    "origin_file":kwargs["file"],
                    "n_rows":kwargs["n_rows"],
                    "col_text":kwargs["col_text"],
                    "col_tags":"labels_default",
                    "scheme":"default",
                    "cat":{"default":kwargs["cat"]},
                    "embeddings":{
                        "sbert":kwargs["sbert"],
                        "fasttext":kwargs["fasttext"],
                    }
                  }
        
        self.schemes.add("default",kwargs["cat"])
        self.schemes.select("default")

        if "col_tags" in kwargs: # if existing tag column
            self.content[self.schemes.col] = self.content[kwargs["col_tags"]]
        else:
            self.content[self.schemes.col] = None

        # Add / compute embeddings as features
        if self.params["embeddings"]["sbert"]:
            self.features.add("sbert",
                              self.compute_embeddings(emb="sbert"))
        if self.params["embeddings"]["fasttext"]:
            self.features.add("fasttext",
                              self.compute_embeddings(emb="fasttext"))

        self.save_params()
        self.save_data()
    
    def compute_embeddings(self,
                           emb:None|str = None):
        """
        Compute embeddings (and save it)
        """
        if emb == "fasttext":
            file = f"{self.name}/fastext"
            if Path(file).exists():
                print("Fasttext embeddings already exist")
                emb_fasttext = pd.read_csv(file,index_col=0)
            else:
                print("Starting to compute fasttext embeddings")
                emb_fasttext = functions.to_fasttext(self.content[self.params["col_text"]])
                emb_fasttext.to_csv(file)
            self.params["embeddings"]["fasttext"] = True
            self.save_params()
            return emb_fasttext
        if emb == "sbert":
            file = f"{self.name}/sbert"
            if Path(file).exists():
                print("Sbert embeddings already exist")
                emb_sbert = pd.read_csv(file,index_col=0)
            else:
                print("Starting to compute sbert embeddings")
                emb_sbert = functions.to_sbert(self.content[self.params["col_text"]])
                emb_sbert.to_csv(file)
            self.params["embeddings"]["sbert"] = True
            self.save_params()
            return emb_sbert
    
    def fit_simplemodel(self,
                        model:str,
                        features:list|str,
                        model_params: None|dict = None
                        ) -> SimpleModel:
        """
        Create and fit a simple model with project data
        """

        # build the dataset with label + predictors

        df_features = self.features.get(features)
        col_label = self.schemes.col
        col_predictors = df_features.columns
        data = pd.concat([self.content[col_label],
                          df_features],
                          axis=1)
        
        s = SimpleModel(model=model,
                        data = data,
                        col_label = col_label,
                        col_predictors = col_predictors,
                        model_params=model_params
                        )
        return s
    
    def update_simplemodel(self, req:dict) -> dict:
        if req["features"] is None or len(req["features"])==0:
            return {"error":"no features"}
        self.simplemodel = self.fit_simplemodel(
                                model=req["current"],
                                features=req["features"],
                                model_params=req["parameters"]
                                )
        return {"success":"new simplemodel"}

    def load_params(self):
        """
        Load YAML configuration file
        """
        with open(f"{self.name}/{self.name}.yaml","r") as f:
            self.params =  yaml.safe_load(f)

        # load also schemes (TO CHANGE IN THE FUTURE)
        self.schemes = Schemes(self.name)
        self.schemes.load({
            "project_name":self.name,
            "name":self.params["scheme"],
            "labels":self.params["cat"][self.params["scheme"]],
            "available":self.params["cat"]
            })

    def save_params(self,params: None|dict=None) -> None:
        """
        Save YAML configuration file
        """
        if params is None:
            params = self.params
        with open(f"{self.name}/{self.name}.yaml", 'w') as f:
            yaml.dump(params, f)

    def update_schemes(self,json: dict) -> None:
        """
        Update schemes from frontend
        """
        self.schemes.load(json)
        self.params["cat"] = self.schemes.available
        self.save_params()

    def load_data(self):
        """
        Load data
        """
        self.content = pd.read_csv(f"{self.name}/{self.name}.csv",
                                   index_col=0,
                                   low_memory=False
                                   )
    def save_data(self):
        """
        Save data
        """
        self.content.to_csv(f"{self.name}/{self.name}.csv")

    def save(self):
        """
        Save all
        """
        # data
        self.content.to_csv(f"{self.name}/{self.name}.csv")
        # params
        self.save_params()
        # features
        self.features.save()
        # simplemodels not saved
    
    def delete_label(self,element_id):
        """
        Delete a recorded tag
        """
        self.content.loc[element_id,self.schemes.col] = None
        return True

    def add_label(self,element_id,label):
        """
        Record a tag
        """
        self.content.loc[element_id,self.schemes.col] = label
        return True

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
        f = self.content[self.schemes.col].isnull()

        if mode == "deterministic": # next row
            element_id = self.content[f].index[0]
        if mode == "random": # random row
            element_id = self.content[f].sample(random_state=42).index[0]
        if mode == "maxprob": # higher prob row
            if self.simplemodel.name is None: # if no model, build default
                print("Build default simple model")
                self.simplemodel = self.fit_simplemodel(model = "liblinear",
                                                        features = "all"
                                                        )
            if tag is None: # default label to first
                tag = self.schemes.labels[0]

            # higher predict value
            element_id = self.simplemodel.proba[f][tag].sort_values(ascending=False).index[0]
        
        # TODO : put a lock on the element when sent ?

        # Pour le moment uniquement l'id et le texte (dans le futur ajouter tous les éléments)
        return  {
                 "element_id":element_id,
                 "content":self.get_element(element_id),
#                 "options":self.get_state()
                }
    
    def get_element(self,element_id):
        """
        Get an element of the database
        """
        columns = ["text"]
        return {"id":element_id,
                "text":self.content.loc[element_id,"text"]
                }


    def get_params(self):
        """
        Send parameters
        """
        return self.params
    
    def get_state(self):
        """
        Send state of the project
        """
        selection_available = ["deterministic","random"]
        if self.simplemodel.name is not None:
            selection_available.append("maxprob")

        options = {
                    
                    "mode":{
                        "available_modes":selection_available,
                        # For the moment mode/on/label are not in the state project side
                        # default values
                        "mode":"deterministic",
                        "on":"untagged",
                        "label":None
                        },
                    "scheme":{
                                "current":self.schemes.name,
                                "available":self.schemes.available
                                },
                    "features":{
                            "available_features":self.features.available
                          }
                   }
        


        
        return  {
                 "type":"state",
                 "content":options
                }
    
    def add_regex(self, name: str, value: str):
        """
        Add regex to features
        """
        if not name in self.features.available:
            pattern = re.compile(value)
            f = self.content[self.params["col_text"]].apply(lambda x: bool(pattern.search(x)))
            self.features.add(name,f)
            print(self.features.available)
            return {"success":"added"}
        else:
            return {"error":"exists already"}

class Features():
    """
    Project features
    No duplicate of data
    TODO : test for the length of the data/same index
    TODO : load available features
    """

    def __init__(self, project_name:str) -> None:
        self.project_name = project_name
        self.available:list = []
        self.map:dict = {}
        self.content = None

    def __repr__(self) -> str:
        return f"Available features : {self.available}"
    
    def save(self):
        """
        Save current state of embeddings
        Temporary : CSV
        """
        metadata = {
            "project_name":self.project_name,
            "available":self.available,
            "map":self.map
        }

        table = self.content.copy()
        table["params"] = None
        table["params"].iloc[0] = json.dumps(metadata)
        table.to_csv(f"{self.project_name}/features.csv")

    def load(self) -> bool:
        """
        Load existing features
        Temporary : CSV
        """
        if Path(f"{self.project_name}/features.csv").exists():
            table = pd.read_csv(f"{self.project_name}/features.csv", 
                                index_col=0,
                                low_memory=False)
            metadata = json.loads(table["params"].iloc[0])
            self.content = table.drop(columns="params").copy()
            del table
            self.map = metadata["map"]
            self.available = metadata["available"]
            self.project_name = metadata["project_name"]
            return True
        else:
            return False

    def add(self, 
            name:str, 
            content:DataFrame|Series) -> None:
        """
        Add feature(s)
        """
        if type(content)==Series:
            content = pd.DataFrame(content)
        content.columns = [f"{name}_{i}" for i in content.columns]
        self.map[name] = list(content.columns)
        # create the dataset
        if self.content is None:
            self.content = content
        else:
            self.content = pd.concat([self.content,content],
                                     axis=1)
        self.available.append(name)

    def delete(self, name:str):
        """
        Delete feature
        """
        if name in self.available:
            col = self.get([name])
            self.available.remove(name)
            self.content.drop(columns=col)
            return {"success":"feature deleted"}
        else:
            return {"error":"feature doesn't exist"}
            
    def get(self,features:list|str = "all"):
        """
        Get specific features
        """
        cols = []
        if features == "all":
            features = self.available

        for i in features:
            if i in self.available:
                cols += self.map[i]
            else:
                print(f"Feature {i} doesn't exist")

        return self.content[cols]
    

class Schemes():
    """
    Project Schemes
    """
    def __init__(self,project_name):
        self.project_name = project_name
        self.name = None
        self.labels = None
        self.col = None
        self.available = {}

    def __repr__(self) -> str:
        return f"Coding schemes available {self.available}"

    def col_name(self):
        return "labels_" + self.name

    def select(self, name):
        if name in self.available:
            self.name = name
            self.labels = self.available[name]
            self.col = self.col_name()
        else:
            raise IndexError

    def add(self, name: str, modalities: list):
        if not name in self.available:
            self.available[name] = modalities
            return True
        else:
            raise IndexError

    def load(self,json):
        """
        Load data
        """
        self.project_name = json["project_name"]
        self.name = json["name"]
        self.labels = json["labels"]
        self.available = json["available"]    
        self.col = self.col_name()    
    
    def dump(self):
        return {
                "project_name":self.project_name,
                "name":self.name,
                "labels":self.labels,
                "available":self.available
                }