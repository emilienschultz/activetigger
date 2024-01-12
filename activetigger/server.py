import os
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

import functions

import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# use Pydantic & BaseModel to type data in API and automatically
# generate the JSON content https://realpython.com/api-integration-in-python/ 

# plus précisémment définir le format des objets qui circulent entre le back et le front
# - element
# - options
# - liste d'éléments

class Server():
    """
    Backend
    """

    def __init__(self):
        """
        Start the server
        """
        self.projects = {}
        logging.warning('Still under development')
        logging.info("Start server")


    def start_project(self,project_name, **kwargs):
        """
        Initialize a project with a project_name
        """
        self.projects[project_name] = Project(project_name, **kwargs)

    # ENDPOINTS (in future FASTAPI)

    def get(self, req):
        """
        Get data from server
        """
        logging.info(f"Get request from frontend {req}")

        # check credentials and parameters

        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        
        p = self.projects[req["project_name"]]

        # deal responses

        if req["type"] == "next" :
            req = p.get_next(req["mode"])
            return req

        if req["type"] == "element" :
            req = p.get_element(req["element_id"])
            return req
        
        if req["type"] == "schemes":
            return {
                    "type":"schemes",
                    "content":p.schemes.dump()
                    }

        #if req["type"] == "params":
        #    return p.params
        
        return {}
    
    def post(self, req):
        """
        Deal post data
        """
        logging.info(f"Post request from frontend {req}")

        # check credentials and parameters

        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        
        p = self.projects[req["project_name"]]

        # manage post request
        if req["type"] == "label" :
            p.add_label(req["content"]["element_id"]["element_id"],req["content"]["label"])
            return {"add":"success"}
        
        if req["type"] == "delete_label":
            p.delete_label(req["content"]["element_id"])
            return {"delete":"success"}            

        if req["type"] == "update_schemes":
            p.update_schemes(req["content"])
            return {"update_schemes":"success"}


class Project():
    """
    Project (database/params)
    """

    def __init__(self, project_name : str, **kwargs):
        """
        Initialize a project (load or create)
        """

        self.name = project_name
        self.params = None
        self.content = None
        self.embeddings = None
        self.schemes = None

        # If project exist
        if self.exists(project_name):
             self.load_params()
             self.load_data()
        # If not
        else:
             self.create(project_name, **kwargs)

        self.compute_embeddings(sbert = self.params["embeddings"]["sbert"],
                                fasttext = self.params["embeddings"]["fasttext"])
    
    def exists(self, project_name):
        return Path(f"{project_name}/{project_name}.yaml").exists()

    def create(self, project_name, **kwargs):
        """
        Create new project
        """

        ## TO DO : tests que tous les paramètres ont bien été renseignés
        
        if not Path(project_name).exists():
            os.makedirs(project_name)
        else:
            print("Erreur, le dossier existe") # gérer la gestion des erreurs

        # Manage parameters
        if not "col_text" in kwargs:
            kwargs["col_text"] = "text"
        if not "sbert" in kwargs:
            kwargs["sbert"] = False
        if not "fasttext" in kwargs:
            kwargs["fasttext"] = False
            
        self.params = {
                    "project_name":project_name,
                    "origin_file":kwargs["file"],
                    "n_rows":kwargs["n_rows"],
                    "cat":{"default":kwargs["cat"]},
                    "col_text":kwargs["col_text"],
                    "embeddings":{
                        "sbert":kwargs["sbert"],
                        "fasttext":kwargs["fasttext"],
                    }
                  }
        
        self.schemes = Schemes(self.name)
        self.schemes.add("default",kwargs["cat"])
        self.schemes.select("default")
        
        self.save_params()

        # Manage data
        self.content = pd.read_csv(kwargs["file"],
                                   index_col=0,
                                   low_memory=False,
                                   nrows=kwargs["n_rows"])
        
        self.content[self.schemes.col] = None
        self.load_predictions()
        self.save_data()

    def load_predictions(self):
        file = f"{self.name}/{self.name}.pred"
        if Path(file).exists():
            proba = pd.read_csv(file,index_col=0)
            self.content["proba"] = proba
        else:
            self.content["proba"] = None
        return True
    
    def compute_embeddings(self, fasttext=False, sbert=False):
        emb = []
        if fasttext:
            file = f"{self.name}/fastext"
            if Path(file).exists():
                print("Fasttext embeddings already exist")
                emb_fasttext = pd.csv(file,index_col=0)
                emb.append(emb_fasttext)
            else:
                emb_fasttext = functions.to_fasttext(self.content[self.params["col_text"]])
                emb_fasttext.to_csv(file)
                emb.append(emb_fasttext)
        if sbert:
            file = f"{self.name}/sbert"
            if Path(file).exists():
                print("Sbert embeddings already exist")
                emb_sbert = pd.csv(file,index_col=0)
                emb.append(emb_sbert)
            else:
                emb_sbert = functions.to_sbert(self.content[self.params["col_text"]])
                emb_sbert.to_csv(file)
                emb.append(emb_sbert)
        self.embeddings = pd.concat(emb)
            
    def __predict_simple_model(self,model=None):
        """
        Fonction temporaire qui génère une prédiction
        """
        self.content["prob"] = np.random.rand(len(self.content))

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
            "name":"default",
            "labels":self.params["cat"]["default"],
            "available":self.params["cat"]
            })

    def save_params(self,params=None):
        """
        Save YAML configuration file
        """
        if params is None:
            params = self.params
        with open(f"{self.name}/{self.name}.yaml", 'w') as f:
            yaml.dump(params, f)

    def update_schemes(self,json):
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
    
    def delete_label(self,element_id):
        self.content.loc[element_id,self.schemes.col] = None
        return True

    def add_label(self,element_id,label):
        self.content.loc[element_id,self.schemes.col] = label
        return True

    def get_next(self,mode = "deterministic"):
        """
        Get next item
        """
        
        # Pour le moment uniquement les cases non nulles
        f = self.content[self.schemes.col].isnull()

        if mode == "deterministic":
            element_id = self.content[f].index[0]
        if mode == "random":
            element_id = self.content[f].sample(random_state=42).index[0]
        if mode == "maxprob":
            element_id = self.content[f].sort_values("prob",ascending=False).index[0]

        # TODO : put a lock on the element when sent ?

        # Pour le moment uniquement l'id et le texte (dans le futur ajouter tous les éléments)
        return  {
                 "element_id":element_id,
                 "content":self.get_element(element_id)
                }
    
    def get_element(self,element_id):
        """
        Get an element of the database
        """
        columns = ["text"]
        return self.content.loc[element_id,columns]

    def get_params(self):
        """
        Send parameters
        """
        return self.params
    
class Schemes():
    """
    Managing project schemes
    """
    def __init__(self,project_name):
        self.project_name = project_name
        self.name = None
        self.labels = None
        self.col = None
        self.available = {}

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