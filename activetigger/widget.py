import ipywidgets as widgets
from ipywidgets import Layout
from IPython.display import display, clear_output
import plotly.graph_objects as go
import json
import requests as rq
from pathlib import Path
import pandas as pd
import time
import asyncio
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import importlib
import numpy as np
from io import BytesIO



class Widget():
    """
    Widget

    Comment
    - only 1 widget sync at the same time in a notebook
    """ 
    async_update = False #stop other sync for widget objects

    def __init__(self, URL_SERVER:str = "http://127.0.0.1:8000") -> None:
        """
        Define general variables
        """
        self.URL_SERVER = URL_SERVER
        self.update_time:int = 2
        self.headers:dict|None = None # authentification with the server
        self.user:str|None = None
        self.project_name: None|str = None
        self.state:dict = {} # global state of the server
        self.current_element:dict|None = None # element to annotate
        self.bert_training:bool = False # if bert is undertraining
        self.history:list = [] # elements annotated during this session
        self.projection_data: pd.DataFrame|str|None = None # get projection data
        self.global_output = widgets.VBox([],layout=Layout(width='100%'))
        self.start()

    def __del__(self): 
        Widget.async_update = False
        print("Widget closed")

    def _post(self,
             route:str, 
             params:dict|None = None, 
             files:str|None = None,
             json_data:dict|None = None,
             data:dict|None = None) -> dict:
        """
        Post to API
        """
        url = self.URL_SERVER + route
        r = rq.post(url, 
                    params = params,
                    json = json_data,
                    data = data,
                    files=files,
                    headers=self.headers)
        
        if r.status_code == 422:
            return {"status":"error", "message":"Not authorized"}

        return json.loads(r.content)
    
    def _get(self,
             route:str, 
             params:dict|None = None, 
             data:dict|None = None,
             is_json = True) -> dict:
        """
        Get from API
        """
        url = self.URL_SERVER + route
        r = rq.get(url, 
                    params = params,
                    data = data,
                    headers=self.headers)
        
        if r.status_code == 422:
            return {"status":"error", "message":"Not authorized"}

        if is_json:
            return json.loads(r.content)
        return r.content
    
    def _connect_user(self, user:str, password:str):
        """
        Connect account and get auth token
        """
        form = {"username":user,
                "password":password}
        r = self._post("/token", data = form)
        if not "access_token" in r:
            print(r)
            return None

        # Update widget configuration
        self.headers = {"Authorization": f"Bearer {r['access_token']}",
           "username":user}
        self.user = user

        # Display
        self.current_user.value = f"Connected as <b>{self.user}</b>"

        # Disable connecting options
        self.existing_users.disabled = True
        self.password.disabled = True
        self.connect_user.disabled = True
        return None

    def start(self) -> None:
        """
        Start project
        """
        # Get existing projects
        try:
            r = self._get("/server")
            existing = r["data"]
        except:
            print(f"Failed to connect to the server. Please check if the server is available at {self.URL_SERVER}")
            return None

        # Stop potential async
        Widget.async_update = False

        # Image
        data_path = importlib.resources.files("activetigger")
        image_path = "img/active_tigger.png"
        img = open(data_path / image_path, 'rb').read()
        img_at = widgets.Image(value=img, format='png', width=50, height=50)

        # Users
        self.existing_users = widgets.Dropdown(description = "User:", 
                                          options = existing["users"],
                                          layout={'width': '200px'})
        self.password = widgets.Text(description = "Password:", layout={'width': '200px'})
        self.connect_user = widgets.Button(description="Connect")
        self.connect_user.style.button_color = 'lightgreen'
        self.connect_user.on_click(lambda x : self._connect_user(user = self.existing_users.value,
                                                 password = self.password.value))
        self.current_user = widgets.HTML(value="Not connected")
        if self.user is not None:
            self.current_user.value = f"Connected as <b>{self.user}</b>"
        # Existing projects
        existing_projects = widgets.Dropdown(
            options=existing["projects"],
            description='Available :',
            layout={'width': '300px'},
            disabled=False)

        # Start existing project
        start = widgets.Button(description="Launch")
        start.style.button_color = 'lightgreen'
        def start_project(b):
            self.project_name = existing_projects.value
            if existing_projects.value is None:
                print("No project selected")
                return None
            self.state = self.get_state()
            if "error" in self.state:
                print("Not connected")
                return None
            self.interface()
        start.on_click(start_project)

        # Create a new project
        create = widgets.Button(description="New project")
        create.on_click(lambda x: self._create_new_project())

        # Delete a project
        delete = widgets.Button(description="Delete project", button_style='danger')
        delete.on_click(lambda x: self._delete_project(existing_projects.value))

        # Display
        clear_output()
        self.output = widgets.VBox([widgets.HBox([img_at, self.existing_users, self.password, self.connect_user]),
                                    self.current_user,
                                    widgets.HBox([existing_projects, start, delete, create]) 
                                    ])
        #display(self.output)
        self.global_output.children = [self.output]
        display(self.global_output)

    def get_state(self) -> dict:
        """
        Get state variable
        """
        state = self._get(route = f"/state/{self.project_name}")
        return state["data"]

    def _delete_project(self, project_name:str) -> dict:
        """
        Delete existing project
        """
        params = {
                "project_name": project_name,
                "user":self.user
                }  
        r = self._post(route = "/projects/delete", 
                       params = params)
        self.start()
        return r

    def _create_new_project(self):
        """
        Create a new project
        """
        clear_output()

        # first step of the panel
        #------------------------

        project_name = widgets.Text(disabled=False,
                                    description="Name:",
                                    layout={'width': '200px'})

        # load file
#        file = widgets.Text(disabled=False,
#                            description="Path:",
#                            layout={'width': '300px'},
#                            value = "path to a csv")
        file = widgets.FileUpload(
            description= 'Select file',
            accept='.csv',
            multiple=False,
        )
        box_file = widgets.HBox([file])
        layout=widgets.Layout(width='100px', margin='0px 0px 0px 50px')
        load = widgets.Button(description="Load",
                              layout=layout)
        load.style.button_color = 'lightgreen'

        # WARNING : BUG dans VS Code sur l'upload donc utiliser un
        # chemin
        #file = widgets.FileUpload(
        #    accept='.csv',
        #    multiple=False
        #)
        #def on_upload_change(change):
        #    print("chargé")
        #    input_file = list(file.value.values())[0]
        #    content = input_file['content']
        #    content = io.StringIO(content.decode('utf-8'))
        #    df = pd.read_csv(content)
        #file.observe(on_upload_change, names='value')
        # nom de la colonne texte
        # nom de la colonne identifiant

        # second step of the panel
        #-------------------------

        # separator
        separate = widgets.HTML(value = "<hr>")

        # select columns
        column_text = widgets.Dropdown(
            options=[],
            description='Text:',
            layout={'width': '200px'},
            disabled=False)

        column_id = widgets.Dropdown(
            options=[],
            description='Id:',
            layout={'width': '200px'},
            disabled=False)
        
        column_label = widgets.Dropdown(
            options=[],
            description='Labels*:',
            layout={'width': '200px'},
            disabled=False)
        
        columns_context = widgets.SelectMultiple(
            options=[],
            description='Context*:',
            layout={'width': '200px'},
            disabled=False)
        
        n_train = widgets.IntText(description = "nrows train", layout={'width': '200px'})
        n_test = widgets.IntText(description = "nrow test", layout={'width': '200px'})

        info = widgets.HTML()

        # Populate
        #---------

        # create the project
        layout=widgets.Layout(width='100px', margin='30px 30px 30px 30px')
        validate = widgets.Button(description="Create",
                              layout=layout)
        def create_project(b):
            data = {
                    "project_name": project_name.value,
                    "user":self.user,
                    "col_text": column_text.value,
                    "col_id":column_id.value,
                    "col_label":column_label.value,
                    "cols_context": list(columns_context.value),
                    "n_train":n_train.value,
                    "n_test":n_test.value
                    }
            files = {'file': (file.value[0]["name"],
                              BytesIO(file.value[0]["content"]))}
            r = self._post(route="/projects/new", 
                       files=files,
                       data=data
                       )
            # if project exit
            print(r)
            if r["status"] == "error":
                print(r["message"])
            else:
                self.start()
        validate.on_click(create_project)
        validate.style.button_color = 'lightgreen'

        # manage 2-level menu display
        self.output = widgets.VBox([widgets.HBox([project_name, box_file, load])])
        
        def load_file(b):
            #df = self._load_file(file.value)
            df = self._load_file(file)
            column_text.options = df.columns
            column_id.options = df.columns
            column_label.options = df.columns
            columns_context.options = df.columns
            info.value = f"Size of the dataset: {len(df)}"
            if len(df.columns)>1:
                column_text.value = column_text.options[1]
                column_id.value = column_id.options[0]
            if len(self.output.children) == 1:
                self.output.children = list(self.output.children) + [separate,
                                                                     info, 
                                                                     column_id,
                                                                     column_text,
                                                                     column_label,
                                                                     columns_context,
                                                                     n_train,
                                                                     n_test,
                                                                     validate]
        load.on_click(load_file)
        self.global_output.children = [self.output]
        display(self.global_output)

    def _load_file(self, file):
        """
        Load file
        """
        if len(file.value) == 0:
            print("File doesn't exist")
            return pd.DataFrame()
        if file.value[0]["type"] != "text/csv":
            print("Only CSV")
            return pd.DataFrame()
        content = file.value[0]["content"]
        df = pd.read_csv(BytesIO(content))
        return df

    # def _load_file(self,path) -> pd.DataFrame:
    #     """
    #     Load file
    #     """
    #     path = Path(path)
    #     if not path.exists():
    #         print("File doesn't exist")
    #         return pd.DataFrame()
    #     if not path.suffix == '.csv':
    #         print("File not csv")
    #         return pd.DataFrame()
    #     df = pd.read_csv(path)
    #     return df
    
    def _display_next(self) -> bool:
        """
        Get next element from the current widget options
        """

        # frame of the visualisation
        try:
            f = self.visualization.children[0]
            x1y1x2y2 = [f['layout']['xaxis']['range'][0], 
                        f['layout']['yaxis']['range'][0],
                        f['layout']['xaxis']['range'][1], 
                        f['layout']['yaxis']['range'][1]]
        except:
            x1y1x2y2 = []

        params = {
                "project_name":self.project_name,
                "scheme":self.select_scheme.value,
                "selection":self._mode_selection.value,
                "sample":self._mode_sample.value,
                "user":self.user,
                "tag":self._mode_label.value,
                "frame":x1y1x2y2
                }
        r = self._get(route = "/elements/next",
                      params = params)
        
        # Managing errors
        if r["status"] == "error":
            print(r["message"])
            return False

        # Update interface
        self.current_element = r["data"]
        self._textarea.value = self.current_element["text"]
        self.info_element.value = str(self.current_element["info"])
        self.info_predict.value = f"Predict SimpleModel: <b>{self.current_element['predict']['label']}</b> (p = {self.current_element['predict']['proba']})"
        return True

    def _display_buttons_labels(self) -> bool:
        """
        Dispaly labels to annotate (panel annotation)
        Managing tag posting
        """
        
        labels = self.state["schemes"]["available"][self.select_scheme.value]

        # function to post
        def send_tag(v):
            data = {
                    "project_name":self.project_name,
                    "scheme":self.select_scheme.value,
                    "element_id":self.current_element["element_id"],
                    "tag":v.description,
                    "user":self.user,
                    "selection":self.current_element["selection"] #mode of selection of the element
                    }
            
            r = self._post(route = "/tags/add",
                            params = {"project_name":self.project_name},
                            json_data = data)
            
            # add in history
            if "error" in r:
                print(r)
            else:
                self.history.append(self.current_element["element_id"])
            self._display_next()

            # check if simplemodel need to be retrained
            if self.is_simplemodel() and (len(self.history) % self.simplemodel_autotrain.value == 0):
                sm = self.state["simplemodel"]["available"][self.user][self.select_scheme.value]
                self.create_simplemodel(self.select_scheme.value,
                           model = sm["name"], 
                           parameters = sm["params"], 
                           features = sm["features"])
                
        # create buttons
        buttons = []
        for t in labels:
            b = widgets.Button(description=t)
            b.on_click(send_tag)
            b.style.button_color = 'lightblue'
            buttons.append(b)
        
        # add buttons
        self._labels.children = buttons
        return True

    def is_simplemodel(self)->bool:
        if self.user in self.state["simplemodel"]["available"]:
            if self.select_scheme.value in self.state["simplemodel"]["available"][self.user]:
                return True
        return False

    def update_global(self):
        """
        Global update of the widget
        """
        self.state = self.get_state()
        self.update_tab_annotations(False)
        self.update_tab_schemes(False)
        self.update_tab_simplemodel(False)
        self.update_tab_description(False)
        self.update_tab_bertmodels(False)
        self.update_tab_features(False)
        self.update_tab_exports(False)

    def update_tab_bertmodels(self, state = True):
        """
        Update BertModel Tab
        """
        if state:
            self.state = self.get_state()

        # possible bertmodels
        self.new_bert_base.options = self.state["bertmodels"]["options"]
        self.new_bert_base.value = self.new_bert_base.options[0]

        # display saved bertmodels for the current scheme (temporary start with _)
        self.available_bert.options = []
        if self.select_scheme.value in self.state["bertmodels"]["available"]:
            self.available_bert.options = self.state["bertmodels"]["available"][self.select_scheme.value].keys()
        self.new_bert_params.value = json.dumps(self.state["bertmodels"]["base_parameters"], indent=2)

        # display status
        self.bert_status.value = f"Currently no model in training"
        if self.bert_training:
            self.bert_status.value = f"Model under training"

        # display start/stop buttons
        if not self.bert_training:
            compute = widgets.Button(description="⚙️Train")
            compute.style.button_color = 'lightgreen'
            compute.on_click(lambda x: self.create_bertmodel())
            self.compute_new_bert.children = [compute]
        else:
            stop = widgets.Button(description="⚙️Stop")
            stop.style.button_color = 'red'
            stop.on_click(lambda x: self.stop_bertmodel())
            self.compute_new_bert.children = [stop]
        return True
        

    def update_tab_features(self, state = True):
        """
        Update Features Tab
        """
        if state:
            self.state = self.get_state()
        self.available_features.options = self.state["features"]["available"]
        if len(self.state["features"]["available"])>0:
            self.available_features.value = self.state["features"]["available"][0]

        self.add_features.options = self.state["features"]["options"].keys()
        self.features_params.value = json.dumps(self.state["features"]["options"][self.add_features.value])

        c = self.state["features"]["training"]
        if len(c) == 0:
            c = "None"
        self.info_features.value = f"Processes currently running: {c}"

        return True

    def update_tab_description(self, state = True):
        """
        Update Description tab
        """
        if state:
            self.state = self.get_state()
        params = {"project_name":self.project_name,
                  "scheme":self.select_scheme.value,
                  "user":self.user}
        r = self._get("/description",params = params)
        text = ""
        for k,v in r["data"].items():
            text += f"<br>- <b>{k}</b>: {v}"
        self.data_description.value = text
        return True

    def update_tab_annotations(self, state = True):
        """
        Update Annotations Tab
        """
        if state:
            self.state = self.get_state()

        self._display_next()
        self._display_buttons_labels()

        self._mode_selection.options = self.state["next"]["methods_min"]
        self._mode_sample.options = self.state["next"]["sample"]
        self._mode_label.disabled = True
        # to display context
        if self.add_context.value:
            self.display_context.value = json.dumps(self.current_element["context"])
        else:
            self.display_context.value = ""
        # case of a simplemodel is available for the user and the scheme
        if self.is_simplemodel():
            self._mode_selection.options = self.state["next"]["methods"]
            self._mode_label.disabled = False
            self._mode_label.options = self.state["schemes"]["available"][self.select_scheme.value]
        # projection
        self.projection_method.options = list(self.state["projections"]["available"].keys())
        self.projection_params.value = json.dumps(self.state["projections"]["available"]["umap"], indent=2)
        self.projection_features.options = self.state["features"]["available"]

    def update_tab_schemes(self, state = True): 
        """
        Update Schemes Tab
        """
        if state:
            self.state = self.get_state()
        self.select_scheme.options = list(self.state["schemes"]["available"].keys())
        self.select_label.options = self.state["schemes"]["available"][self.select_scheme.value]

    def update_tab_simplemodel(self, state = True):
        """
        Update Simplemodel Tab
        """
        if state:
            self.state = self.get_state()

        self.select_simplemodel.options = list(self.state["simplemodel"]["options"].keys())
        self.select_features.options = self.state["features"]["available"]

        # if a model has already be trained for the user and the scheme
        if (self.user in self.state["simplemodel"]["available"]) and (self.select_scheme.value in self.state["simplemodel"]["available"][self.user]):
            current_model = self.state["simplemodel"]["available"][self.user][self.select_scheme.value]
            name = current_model['name']
            statistics = f"F1: {round(current_model['statistics']['weighted_f1'],2)} - accuracy: {round(current_model['statistics']['accuracy'],2)}"
            self.simplemodel_params.value = json.dumps(current_model["params"], indent=2)
            self.select_simplemodel.value = name
        else:
            name = "No model"
            statistics = ""
            self.simplemodel_params.value = ""

        # display information
        self.simplemodel_state.value = f"Current model: {name}"
        self.simplemodel_statistics.value = statistics

    def update_tab_data(self, state = True):
        """
        Update Data tab
        """
        if state:
            self.state = self.get_state()
        params = {
                   "project_name":self.project_name,
                   "scheme":self.select_scheme.value,
                   "min":self.sample_min.value,
                   "max":self.sample_max.value,
                   "mode":self.sample_type.value
                  }
        r = self._get("/elements/table", params = params)
        df = pd.DataFrame(r["data"])
        buttons = []
        for i,j in df.iterrows():
            options = self.state["schemes"]["available"][self.select_scheme.value]
            if not j["labels"] in options:
                options.append(j["labels"]) # case of a old label
            menu = widgets.Dropdown(options = options, 
                                 value = j["labels"],
                                 layout = {"width":"100px"})
            menu.layout.id = i
            buttons.append(widgets.HBox([
                
                widgets.HTML(value=f"<hr>{i} - <small>{j['text']}</small>",
                             layout = {"width":"500px"}),
                menu, # important to keep at the end, used for sending the table
            ]))
        self.display_table.children = buttons

    def update_tab_exports(self, state = True):
        if state:
            self.state = self.get_state()
        self.export_features_columns.options = self.state["features"]["available"]

        prediction = []
        if self.select_scheme.value in self.state["bertmodels"]["available"]:
            prediction = [i[0] for i in self.state["bertmodels"]["available"][self.select_scheme.value] if i[1]] #if predict
        self.select_bert_model_predict.options = prediction

        bert = []
        if self.select_scheme.value in self.state["bertmodels"]["available"]:
            bert = [i for i in self.state["bertmodels"]["available"][self.select_scheme.value].keys()] #if compressed
        self.select_bert_model.options  = bert

    def create_scheme(self, s):
        """
        Create new scheme
        """
        if s == "":
            return "Empty"
        params = {"project_name":self.project_name}
        data = {
                "project_name":self.project_name,
                "name":s,
                "tags":[],
                #"user":self.user
                }
        r = self._post("/schemes/add", 
                       params = params, 
                       json_data = data)
        self.update_tab_schemes()
        return r
    
    def delete_scheme(self, s):
        """
        Delete scheme
        """
        if s == "":
            return "Empty"
        params = {"project_name":self.project_name}
        data = {
                "project_name":self.project_name,
                "name":s,
                #"tags":[],
                #"user":self.user
                }
        r = self._post("/schemes/delete", 
                       params = params, 
                       json_data = data)
        self.update_tab_schemes()
        return r
    
    def delete_label(self, label:str):
        """
        Delete label in a scheme
        """
        if label == "":
            return "Empty"
        params = {"project_name":self.project_name,
                  "scheme":self.select_scheme.value,
                  "label":label,
                  "user":self.user}
        r = self._post("/schemes/label/delete", 
                       params = params)
        self.update_tab_schemes()
        return r
    
    def _delete_bert(self, bert_name):
        """
        Delete bert model
        """
        params = {"project_name":self.project_name,
                  "bert_name":bert_name,
                  "user":self.user
                }
        r = self._post("/models/bert/delete", 
                       params = params)
        self.update_tab_bertmodels()
        return r

    def create_label(self, text_field):
        """
        Create label in a scheme
        """
        label = text_field.value
        if label == "":
            return "Empty"
        params = {"project_name":self.project_name,
                  "scheme":self.select_scheme.value,
                  "label":label,
                  "user":self.user}
        r = self._post("/schemes/label/add", 
                       params = params)
        self.update_tab_schemes()
        text_field.value = ""
        return r
    
    def create_simplemodel(self, 
                           scheme:str, 
                           model:str, 
                           parameters:str, 
                           features:list):
        """
        Create a simplemodel
        """
        if model is None:
            return "Model missing"
        if parameters is None:
            return "Parameters missing"
        if (features is None) or (len(features)==0):
            return "Need at least one feature" 
        # TODO : test if parameters is valid
        params = {"project_name":self.project_name}
        if type(parameters) is str:
            try:
                parameters = json.loads(parameters)
            except:
                print(parameters)
                raise ValueError("Problem in the json parameters")
        data = {
                "model":model,
                "features":features,
                "params":parameters,
                "scheme":scheme,
                "user":self.user
                }
        
        r = self._post("/models/simplemodel", 
                       params = params, 
                       json_data = data)
        self.update_tab_simplemodel()
        return True
    
    def create_bertmodel(self):
        """
        Create a bertmodel
        """
        self.compute_new_bert.disabled = True
        if self.new_bert_base.value is None:
            return "Model missing"
        if self.new_bert_params.value is None:
            return "Parameters missing"
        try:
            bert_params = json.loads(self.new_bert_params.value)
        except:
            raise ValueError("Problem in the json parameters")
        
        params = {"project_name":self.project_name}
        data = {
                "project_name":self.project_name,
                "scheme":self.select_scheme.value,
                "user":self.user,
                "name":f"_{self.user}", # générique
                "base_model":self.new_bert_base.value,
                "params":bert_params,
                "test_size":0.2
                }
        
        r = self._post("/models/bert/train", 
                       params = params, 
                       json_data = data)
        time.sleep(2)
        self.bert_training = True
        self.update_tab_bertmodels()
        self.compute_new_bert.disabled = False
        return True
    
    def _start_bert_prediction(self, model_name:str):
        """
        Start prediction
        """
        params = {"project_name":self.project_name,
                  "user":self.user,
                  "model_name":model_name
                  }
        r = self._post("/models/bert/predict", 
                params = params)
        if "error" in r:
            print(r)
        print(r)
        self.compute_prediction.disabled = True
        return True

    def stop_bertmodel(self):
        """
        Stop bertmodel training
        """
        self.compute_new_bert.disabled = True
        params = {"project_name":self.project_name,
                  "user":self.user}
        r = self._post("/models/bert/stop", 
                params = params)
        time.sleep(2)
        self.bert_training = False
        self.compute_new_bert.disabled = False
        self.update_tab_bertmodels()
        return True

    def _display_element(self, element_id):
        """
        Display specific element
        """
        r = self._get(route = f"/elements/{element_id}",
                      params = {"project_name":self.project_name,
                                "scheme":self.select_scheme.value})
        # Managing errors
        if r["status"]=="error":
            print(r)
            return False
        # Update interface
        self.current_element = r["data"]
        self._textarea.value = self.current_element ["text"]
        return True

    def _get_previous_element(self) -> bool:
        """
        Load previous element in history
        """
        if len(self.history) == 0:
            print("No element in history")
            return False
        element_id = self.history.pop()
        r = self._display_element(element_id) 
        return r
    
    def compute_feature(self, feature_name, feature_params) -> bool:
        """
        Compute feature
        """
        if not feature_name in self.state["features"]["options"].keys():
            return "This feature doesn't exist"
        try:
            feature_params = json.loads(feature_params) #test if ok
        except:
            raise ValueError("Problem in the json parameters")
        r = self._post(f"/features/add/{feature_name}", 
                    params ={
                            "project_name":self.project_name,
                            "user":self.user
                            },
                    json_data = {"params":feature_params}
                    )
        self.update_tab_features()
        return True
    
    def delete_feature(self, feature_name) -> bool:
        """
        Delete existing feature
        """
        r = self._post(f"/features/delete/{feature_name}", 
                    params = {"project_name":self.project_name,
                              "user":self.user})
        self.update_tab_features()
        return True
    
    def add_regex(self, value:str, name:str|None = None) -> bool:
        """
        Add regex as feature
        """
        if name is None:
            name = value
        
        name = f"regex_{self.user}_{name}"

        data = {
            "project_name":self.project_name,
            "name":name,
            "value":value,
            "user":self.user
            }
        
        r = self._post("/features/add/regex",
            params = {"project_name":self.project_name},
            json_data=data)
        self.update_tab_features()
        return True
    
    def save_bert(self, former_name, new_name):
        params = {"project_name":self.project_name,
                  "former_name":former_name,
                  "new_name":new_name,
                  "user":self.user
                  }
        r = self._post("/models/bert/rename",
            params = params)
        return r
        
    def export_data(self, format):
        """
        Get exported data
        """
        params = {"project_name":self.project_name,
                  "scheme":self.select_scheme.value, #current scheme
                  "format":format
                  }
        r = self._get("/export/data",
            params = params,
            is_json= False)
        with open(f"./data_export.{format}","wb") as f:
            f.write(r)
        print(f"data exported in './data_export.{format}'")
        return True
    
    def export_features(self, features:list, format:str):
        """
        Get exported features
        """
        params = {"project_name":self.project_name,
                  "features":features,
                  "format":format
                  }
        r = self._get("/export/features",
            params = params,
            is_json= False)
        with open(f"./features_export.{format}","wb") as f:
            f.write(r)
        print(f"Features exported in './features_export.{format}'")
        return True
    
    def export_prediction(self, name:str, format:str):
        """
        Get exported prediction for a BERT model
        """
        params = {"project_name":self.project_name,
                  "name":name,
                  "format":format
                  }
        r = self._get("/export/prediction",
            params = params,
            is_json= False)
        with open(f"./prediction_export.{format}","wb") as f:
            f.write(r)
        print(f"Prediction exported in './prediction_{name}_export.{format}'")

    def export_bert(self, name:str):
        """
        Get BERT Model
        """
        if name is None:
            return None
        params = {"project_name":self.project_name,
                  "name":name
                  }
        r = self._get("/export/bert",
            params = params,
            is_json= False)
        if type(r) is dict:
            print(r)
            return None
        with open(f"./{name}.tar.gz","wb") as f:
            f.write(r)
        print(f"Bert exported in './{name}.tar.gz'")

    def compute_projection(self):
        """
        Start computing projection
        """
        params = {
                "project_name":self.project_name,
                "user":self.user
                }
        
        try:
            proj_params = json.loads(self.projection_params.value)
        except:
            raise ValueError("Problem in the json parameters")

        data = {
            "method":self.projection_method.value, 
            "features":self.projection_features.value,
            "params":proj_params,
            }
        r = self._post("/elements/projection/compute",
            params = params,
            json_data = data)
        if r["status"] == "waiting":
            self.projection_data = "computing"
            self.visualization.children = [widgets.HTML(value = self.projection_data)]
        else:
            print(r)
    

    def plot_visualisation(self):
        """
        Produce the visualisation for the projection
        """
        df = self.projection_data
        df["to_show"] = df.apply(lambda x : f"{x.name}| {x['texts'][0:100]}...", axis=1)
        f = go.FigureWidget([go.Scatter(x=df[df["labels"]==i]["0"], 
                                        y=df[df["labels"]==i]["1"], 
                                        mode='markers', 
                                        name = i,
                                        customdata = np.stack((df[df["labels"]==i]["to_show"],), axis=-1),
                                        hovertemplate="%{customdata[0]}",
                                        showlegend = True) for i in df["labels"].unique()])
        f.layout.hovermode = 'closest'
        def update_point(trace, points, selector):
            # select specific text
            if len(points.point_inds)>0:
                element_id = trace.customdata[points.point_inds][0][0].split("|")[0] #TODO améliorer
                print(element_id)
                self._display_element(element_id)
        for i in range(0,len(f.data)):
            f.data[i].on_click(update_point)
        self.visualization.children = [f]

    def display_bert_informations(self, name):
        """
        Display statistics for a BERT Model
        """
        params = {"project_name":self.project_name,
                            "name":name}
        r = self._get("/models/bert", params = params)
        if r["status"] == "error":
            print(r)
            return 
        loss = pd.DataFrame(r["data"]["loss"])
        with self.bert_statistics:
            clear_output(wait=True)
            print(json.dumps(r["data"]["parameters"],indent=2))
            fig, ax = plt.subplots(figsize=(3,2))
            fig = loss.plot(ax = ax)
            plt.show(fig)
            if "f1" in r["data"]:
                print("f1:",r["data"]["f1"])
                print("precision:",r["data"]["precision"])
                print("recall:",r["data"]["recall"])
            else:
                print("Compute prediction for scores")
    
    def get_projection_data(self):
        """
        Get projection data
        """
        params = {
                "project_name":self.project_name, 
                "user":self.user,
                "scheme":self.select_scheme.value
                }
        r = self._get("/elements/projection/current",
            params = params)
        return r

    async def update_state(self):
        """
        Async function to update state for all long term processes
        - check bertmodels
        - check simplemodels
        - check features
        - check projections
        """
        while Widget.async_update:
            self.state = self.get_state()
            await asyncio.sleep(self.update_time)
            if len(self.state) == 0:
                continue
            # check bertmodel status
            if self.bert_training and (not self.user in self.state["bertmodels"]["training"]):
                self.bert_training = False
                self.update_tab_bertmodels(state=False)
            # check features status
            if self.state["features"]["available"] != self.available_features.options:
                self.available_features.options = self.state["features"]["available"]
            # check projection status
            if (type(self.projection_data) is str) and (self.projection_data == "computing"):
                r = self.get_projection_data()
                if ("data" in r) and (type(r["data"]) is dict):
                    print("get projection data")
                    self.projection_data = pd.DataFrame(r["data"])
                    self.plot_visualisation()

    def interface(self):
        """
        General interface
        - divided by tab
        """

        # updating thread
        #for task in asyncio.all_tasks():
        #    task.cancel()
        Widget.async_update = True
        asyncio.create_task(self.update_state())

        #------------
        # Tab schemes
        #------------
        self.select_scheme = widgets.Dropdown(description="Select: ", value="", options=[""])
        valid_delete_scheme = widgets.Button(description = "Delete", button_style = "danger")
        valid_delete_scheme.on_click(lambda b : self.delete_scheme(self.select_scheme.value))
        new_scheme = widgets.Text(description="New: ")
        valid_new_scheme = widgets.Button(description = "Create")
        valid_new_scheme.style.button_color = 'lightgreen'
        valid_new_scheme.on_click(lambda b : self.create_scheme(new_scheme.value))
        self.select_label = widgets.Dropdown(description="Labels: ")
        valid_delete_label = widgets.Button(description = "Delete", button_style = "danger")
        valid_delete_label.on_click(lambda b : self.delete_label(self.select_label.value))
        new_label = widgets.Text(description="New label: ")
        valid_new_label = widgets.Button(description = "Create")
        valid_new_label.style.button_color = 'lightgreen'
        valid_new_label.on_click(lambda b : self.create_label(new_label))

        # Populate
        self.update_tab_schemes()
        if len(self.select_scheme.options)>0:
            self.select_scheme.value = self.select_scheme.options[0]
        if len(self.select_label.options)>0:
            self.select_label.value = self.select_label.options[0]
        # change labels if scheme change
        def on_change_scheme(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.update_tab_schemes()
        self.select_scheme.observe(on_change_scheme)

        # Group in tab
        tab_schemes = widgets.VBox([
                            widgets.HBox([self.select_scheme, valid_delete_scheme]),
                            widgets.HBox([new_scheme, valid_new_scheme]),
                            widgets.HTML(value="<hr>"),
                            widgets.HBox([self.select_label, valid_delete_label]),
                            widgets.HBox([new_label, valid_new_label]),
                        ])

        #-------------
        # Tab Features
        #-------------
        self.info_features  = widgets.HTML(value = "No process currently running")
        self.available_features =  widgets.Dropdown(description = "Available")
        delete_feature = widgets.Button(description = "Delete", button_style="danger")
        delete_feature.on_click(lambda x: self.delete_feature(self.available_features.value))
        self.add_features = widgets.Dropdown(description="Add: ", value="", options=[""])
        self.features_params = widgets.Textarea(layout=widgets.Layout(width='200px',height='100px'))
        def if_change_feature(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.features_params.value = json.dumps(self.state["features"]["options"][self.add_features.value],
                                                        indent = 2)
        self.add_features.observe(if_change_feature)
        valid_compute_features = widgets.Button(description = "⚙️Compute")
        valid_compute_features.on_click(lambda x : self.compute_feature(self.add_features.value, 
                                                                        self.features_params.value))
        valid_compute_features.style.button_color = 'lightgreen'
        add_regex_value = widgets.Text(description="Add regex:")
        valid_regex = widgets.Button(description = "Add")
        valid_regex.style.button_color = 'lightgreen'
        valid_regex.on_click(lambda x: self.add_regex(add_regex_value.value))

        # Populate
        self.update_tab_features()

        # Group in tab
        tab_features = widgets.VBox([
            widgets.HBox([self.available_features,delete_feature]),
            widgets.HTML(value = "<hr>"),
            widgets.HBox([self.add_features,self.features_params,valid_compute_features]),
            widgets.HTML(value = "<hr>"),
            widgets.HBox([add_regex_value,valid_regex]),
            self.info_features,
             ])

        #-----------
        # Tab codage
        #-----------
        self._textarea = widgets.Textarea(value="",
                                   layout=widgets.Layout(width='600px',height='150px'), 
                                   description='')
        self._back = widgets.Button(description = "◄ back",layout=widgets.Layout(width='100px'))
        self._back.on_click(lambda x : self._get_previous_element())
        self._mode_selection = widgets.Dropdown(layout=widgets.Layout(width='120px'))
        def on_change_method(change): # if method change, change the current element
            if change['type'] == 'change' and change['name'] == 'value':
                self._display_next()
        self._mode_selection.observe(on_change_method)
        self._mode_sample = widgets.Dropdown(layout=widgets.Layout(width='120px'))
        self._mode_label = widgets.Dropdown(layout=widgets.Layout(width='120px'),
                                            disabled=True)
        self._labels = widgets.HBox()
        self.info_element = widgets.HTML()
        self.info_predict = widgets.HTML()
        self.add_context = widgets.ToggleButton(description='Context', value=False, icon='valid')
        def update_context(x):
            self.update_tab_annotations()
        self.add_context.observe(update_context, "value")
        self.display_context = widgets.HTML()

        # Part projection visualisation
        self.projection_method = widgets.Dropdown(description = "Method")
        def on_change_method(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.projection_params.value = json.dumps(self.state["projections"]["available"][self.projection_method.value], indent=2)
        self.projection_method.observe(on_change_method)
        self.projection_params = widgets.Textarea(layout = widgets.Layout(width='200px',height='100px'))
        self.projection_features = widgets.SelectMultiple(description = "Features")
        self.projection_compute = widgets.Button(description = "Visualize")
        self.projection_compute.on_click(lambda x: self.compute_projection())
        self.visualization = widgets.HBox([])
        self.projection = widgets.Accordion(children=[widgets.VBox([
                                    widgets.HBox([self.projection_method, self.projection_compute]),
                                    widgets.HBox([self.projection_features,self.projection_params]),
                                                  self.visualization
                                                                    ]
                                                     )], 
                                            titles=('Projection',))
        
        

        # Populate
        self.update_tab_annotations()
        self._mode_selection.value = self._mode_selection.options[0]
        self._mode_sample.value = self._mode_sample.options[0]

        # Group in tab
        tab_annotate = widgets.VBox([
                            #self._schemes,
                             widgets.HBox([self._back,
                                    self._mode_selection,
                                    self._mode_sample,
                                    self._mode_label,
                                    self.add_context]),
                              self._textarea,
                              self.info_predict,
                              self.info_element,
                              self.display_context,
                              self._labels,
                              widgets.HTML("<hr>"),
                              self.projection
                            ])

        #---------
        # Tab data
        #---------
        self.sample_type = widgets.Dropdown(description="On: ", value="recent", 
                                            options=["all","tagged","untagged","recent"],
                                            layout={'width': '200px'})
        self.sample_min = widgets.IntText(value=0, description='Min:', disabled=False, layout={'width': '200px'})
        self.sample_max = widgets.IntText(value=0, description='Max:', disabled=False, layout={'width': '200px'})
        self.display_table = widgets.VBox()
        valid_sample = widgets.Button(description = "Get")
        valid_sample.on_click(lambda b : self.update_tab_data())
        valid_sample.style.button_color = 'lightblue'
        modify_table = widgets.Button(description = "Valid changes")
        modify_table.style.button_color = 'orange'

        def send_table():
            data = {
                "scheme":self.select_scheme.value,
                "list_ids":[i.children[-1].layout.id for i in self.display_table.children],
                "list_labels":[i.children[-1].value for i in self.display_table.children]
            }
            r = self._post("/elements/table", 
                       json_data = data, 
                       params = {"project_name":self.project_name,
                                 "user":self.user
                                 })
        modify_table.on_click(lambda b : send_table())

        # Populate
        self.sample_min.value = 0
        self.sample_max.value = 10
        self.sample_type.value = "all"
        self.update_tab_data()

        # Group in tab
        tab_data = widgets.VBox([widgets.HBox([
                                    self.sample_type, 
                                    self.sample_min, 
                                    self.sample_max, 
                                    valid_sample
                                    ]),
                                 self.display_table,
                                 modify_table
                                  ])

        #---------------
        # Tab statistics
        #---------------
        self.data_description = widgets.HTML(layout={'width': '300px', 'height':'200px'})

        # Populate
        self.update_tab_description()

        # Group in tab
        tab_description = widgets.VBox([self.data_description])

        #----------------
        # Tab SimpleModel
        #----------------
        self.simplemodel_state = widgets.HTML(value = "State")
        self.simplemodel_statistics= widgets.HTML(value = "Statistics")

        self.select_simplemodel =  widgets.Dropdown(description = "Models")
        def on_change_scheme(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.simplemodel_params.value = json.dumps(self.state["simplemodel"]["options"][self.select_simplemodel.value],
                                                           indent=2)
        self.select_simplemodel.observe(on_change_scheme)
        self.select_features = widgets.SelectMultiple()
        self.simplemodel_params = widgets.Textarea(value=json.dumps({}),
                                                   layout=widgets.Layout(width='300px',height='200px'))
        valid_model = widgets.Button(description = "⚙️Train")
        valid_model.on_click(lambda b : self.create_simplemodel(scheme=self.select_scheme.value, #attention il faudra revoir le choix du scheme
                                                               model = self.select_simplemodel.value,
                                                               parameters = self.simplemodel_params.value,
                                                               features = self.select_features.value))
        self.simplemodel_autotrain = widgets.IntSlider(min=1, max=50, 
                                      description="")
        # Populate
        self.update_tab_simplemodel()

        # Group in tab
        tab_simplemodel = widgets.VBox([
                            widgets.HBox([self.simplemodel_state,self.simplemodel_statistics]),
                            self.select_simplemodel,
                            widgets.HBox([
                                          self.select_features,
                                          self.simplemodel_params
                                          ]),
                            valid_model,
                            widgets.HTML(value="Autotrain every:"),
                            self.simplemodel_autotrain
             ])

        #--------------
        # Tab BertModel
        #--------------

        """
        Logic : train a bertmodel by user
        which can be saved in a specific name
        TODO : upgrade management of models
        """

        self.bert_status = widgets.Text(disabled=True)
        self.available_bert = widgets.Dropdown(description="Select:")
        self.bert_statistics = widgets.Output()
        def on_change_model(change): # if select one, display its options on_select
            if change['type'] == 'change' and change['name'] == 'value':
                # available predict button
                if self.state["bertmodels"]["available"][self.select_scheme.value][self.available_bert.value][0]:
                    self.compute_prediction.disabled = True
                else:
                    self.compute_prediction.disabled = False
                # get information about the model
                self.display_bert_informations(self.available_bert.value)
        self.available_bert.observe(on_change_model)
        self.bert_summary = widgets.Accordion(children=[self.bert_statistics], 
                                            titles=('Description',))

        self.compute_prediction = widgets.Button(description = "Compute prediction", disabled = True)
        self.compute_prediction.on_click(lambda x : self._start_bert_prediction(self.available_bert.value))
        self.new_bert_base = widgets.Dropdown(description="Base:")
        self.new_bert_params = widgets.Textarea(layout={'width': '200px','height':"200px"})
        self.compute_new_bert = widgets.VBox()
        delete_bert = widgets.Button(description = "Delete Bert", button_style = "danger")
        delete_bert.on_click(lambda x : self._delete_bert(self.available_bert.value))
        self.bert_name = widgets.Text(description="Name:", layout={'width': '150px'}, value="Name")
        self.record_bert = widgets.Button(description = "Save Bert")
        self.record_bert.on_click(lambda x : self.save_bert(self.available_bert.value, 
                                                            self.bert_name.value))

        # Populate
        self.update_tab_bertmodels()

        # Group in tab
        tab_bertmodel = widgets.VBox([
                                self.bert_status,
                                widgets.HBox([self.available_bert,delete_bert, self.compute_prediction]),
                                self.bert_summary,
                                widgets.HTML(value="<hr>Train new bert<br>"),
                                self.new_bert_base,
                                self.new_bert_params,
                                self.compute_new_bert,
                                widgets.HTML(value="<hr>Save current model<br>"),
                                widgets.HBox([self.bert_name, self.record_bert]),
                        
                             ])

        #-----------
        # Tab Export
        #-----------

        layout_button=widgets.Layout(width='80px')
        layout_menu=widgets.Layout(width='150px')


        # data
        export_tagged_data_presentation = widgets.HTML(value="<hr>Export tagged data<br>")
        #export_tagged_data_columns = widgets.SelectMultiple(layout = layout_menu)
        export_tagged_data_format = widgets.Dropdown(options = ["csv","parquet"], layout = layout_button)
        valid_export_tagged_data = widgets.Button(description = "⬇", layout = layout_button)
        valid_export_tagged_data.on_click(lambda x: self.export_data(format = export_tagged_data_format.value))

        # embeddings
        export_features_presentation = widgets.HTML(value="<hr>Export embeddings<br>")
        self.export_features_columns = widgets.SelectMultiple(layout = layout_menu)
        export_features_format = widgets.Dropdown(options = ["csv","parquet"], layout = layout_button)
        valid_export_features = widgets.Button(description = "⬇", layout = layout_button)
        valid_export_features.on_click(lambda x: self.export_features(features = self.export_features_columns.value,
                                                                  format = export_features_format.value))

        # bert predictions
        export_predictions_presentation = widgets.HTML(value="<hr>Export BERT predictions<br>")
        self.select_bert_model_predict = widgets.Dropdown(layout = layout_menu)
        export_bert_format = widgets.Dropdown(options = ["csv","parquet"], layout = layout_button)
        valid_export_predict = widgets.Button(description = "⬇", layout = layout_button)
        valid_export_predict.on_click(lambda x: self.export_prediction(name = self.select_bert_model_predict.value,
                                                                  format = export_features_format.value))

        # bert models
        export_bert_presentation =  widgets.HTML(value="<hr>Export BERT models<br>")
        self.select_bert_model = widgets.Dropdown(layout = layout_menu)
        valid_export_bertmodel = widgets.Button(description = "⬇", layout = layout_button)
        valid_export_bertmodel.on_click(lambda x: self.export_bert(name = self.select_bert_model.value))

        tab_export = widgets.VBox([
            export_tagged_data_presentation, widgets.HBox([
                          #export_tagged_data_columns,
                          export_tagged_data_format,
                          valid_export_tagged_data]),
            export_features_presentation, widgets.HBox([
                          self.export_features_columns,
                          export_features_format,
                          valid_export_features
                          ]),
            export_predictions_presentation, widgets.HBox([
                          self.select_bert_model_predict,
                          export_bert_format,
                          valid_export_predict
                        ]),
            export_bert_presentation, widgets.HBox([
                          self.select_bert_model,
                          valid_export_bertmodel
                        ]),
        ])


        # Populate
        self.update_tab_exports()

        # display global widget
        self.output = widgets.Tab([tab_schemes,
                                   tab_features,
                                   tab_annotate,
                                   tab_description,
                                   tab_data,
                                   tab_simplemodel,
                                   tab_bertmodel,
                                   tab_export],
                                  titles = [
                                            "Schemes",
                                            "Features",
                                            "Annotate",
                                            "Description",
                                            "Data",
                                            "SimpleModels",
                                            "BertModels",
                                            "Export"
                                            ])
        
        # Update everything on tab change
        def on_tab_selected(change):
            self.update_global()
        self.output.observe(on_tab_selected, names='selected_index')

        # Display
        general_info = widgets.HTML(value=f"<b>Current user: {self.user}</b>")
        self.global_output.children = [general_info, self.output]
        