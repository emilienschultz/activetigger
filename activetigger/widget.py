import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import requests as rq
from pathlib import Path
import pandas as pd
import time
import asyncio

URL_SERVER = "http://127.0.0.1:8000"
headers = {'x-token': 'your_token'}

class Widget():
    """
    Widget
    """
    def __init__(self) -> None:
        """
        Define general variables
        """
        self.update_time:int = 2
        self.user:str = "local"
        self.project_name: None|str = None
        self.current_element:dict|None = None
        self.bert_training:bool = False
        self.history:list = []
        self.state:dict = {}
        self.start()

    def _post(self,
             route:str, 
             params:dict|None = None, 
             files:str|None = None,
             json_data:dict|None = None,
             data:dict|None = None) -> dict:
        """
        Post to API
        """
        url = URL_SERVER + route
        r = rq.post(url, 
                    params = params,
                    json = json_data,
                    data = data,
                    files=files,
                    headers=headers)
        return json.loads(r.content)
    
    def _get(self,
             route:str, 
             params:dict|None = None, 
             data:dict|None = None) -> dict:
        """
        Get from API
        """
        url = URL_SERVER + route
        r = rq.get(url, 
                    params = params,
                    data = data,
                    headers=headers)
        return json.loads(r.content)

    def start(self) -> None:
        """
        Start project
        """
        # Get existing projects
        existing = self._get("/projects")

        # Image
        image_path = "../img/active_tigger.png"
        img = open(image_path, 'rb').read()
        img_at = widgets.Image(value=img, format='png', width=50, height=50)

        # Existing projects
        existing_projects = widgets.Dropdown(
            options=existing["existing projects"],
            description='Available :',
            layout={'width': '300px'},
            disabled=False)

        # Start existing project
        start = widgets.Button(description="Connect")
        start.style.button_color = 'lightgreen'
        def start_project(b):
            self.project_name = existing_projects.value
            self.state = self.get_state()
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
        self.output = widgets.HBox([img_at, 
                                    existing_projects, 
                                    start, 
                                    delete, 
                                    create])
        display(self.output)

    def get_state(self) -> dict:
        """
        Get state variable
        """
        state = self._get(route = f"/state/{self.project_name}")
        return state

    def _delete_project(self, project_name:str) -> dict:
        """
        Delete existing project
        """
        params = {"project_name": project_name}
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
        file = widgets.Text(disabled=False,
                            description="Path:",
                            layout={'width': '300px'},
                            value = "path to a csv")
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
                    "cols_context":columns_context.value,
                    "n_train":n_train.value,
                    "n_test":n_test.value
                    }
            files = {'file': (file.value,
                              open(file.value, 'rb'))}
            r = self._post(route="/projects/new", 
                       files=files,
                       data=data
                       )
            # if project exit
            if "error" in r:
                print("Project name alreay exists")
            else:
                self.start()
        validate.on_click(create_project)
        validate.style.button_color = 'lightgreen'

        # manage 2-level menu display
        self.output = widgets.VBox([widgets.HBox([project_name, file, load])])
        
        def load_file(b):
            df = self._load_file(file.value)
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
        display(self.output)

    def _load_file(self,path) -> pd.DataFrame:
        """
        Load file
        """
        path = Path(path)
        if not path.exists():
            print("File doesn't exist")
            return pd.DataFrame()
        if not path.suffix == '.csv':
            print("File not csv")
            return pd.DataFrame()
        df = pd.read_csv(path)
        return df
    
    def _display_next(self) -> bool:
        """
        Get next element from the current widget options
        """
        params = {
                          "project_name":self.project_name,
                          "scheme":self.select_scheme.value,
                          "selection":self._mode_selection.value,
                          "sample":self._mode_sample.value,
                          "user":self.user,
                          "tag":None
                      }
        r = self._get(route = "/elements/next",
                      params = params)
        
        # Managing errors
        if "error" in r:
            print(r)
            return False

        # Update interface
        self.current_element = r
        self._textarea.value = r["text"]

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
                    }
            r = self._post(route = "/tags/add",
                       params = {"project_name":self.project_name},
                       json_data = data)
            # add in history
            self.history.append(self.current_element["element_id"])
            if "error" in r:
                print(r)
            self._display_next()

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

    def update_global(self):
        """
        Global update of the widget
        """
        self.state = self.get_state()
        self._display_buttons_labels()
        self.update_tab_annotations(False)
        self.update_tab_schemes(False)
        self.update_tab_simplemodel(False)
        self.update_tab_description(False)
        self.update_tab_bertmodels(False)
        self.update_tab_features(False)

    def update_tab_bertmodels(self, state = True):
        """
        Update BertModel Tab
        """
        if state:
            self.state = self.get_state()

        # possible bertmodels
        self.new_bert_base.options = self.state["bertmodels"]["options"]
        self.new_bert_base.value = self.new_bert_base.options[0]

        # display bertmodels for the current scheme
        if self.select_scheme.value in self.state["bertmodels"]["available"]:
            self.available_bert.options = self.state["bertmodels"]["available"][self.select_scheme.value]
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

        self.add_features.options = self.state["features"]["options"]

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
        r = self._get("/elements/stats",params = params)
        self.data_description.value = json.dumps(r,indent=2)
        return True

    def update_tab_annotations(self, state = True):
        """
        Update Annotations Tab
        """
        if state:
            self.state = self.get_state()
        self._mode_selection.options = ["deterministic","random"]
        self._mode_sample.options = self.state["next"]["sample"]
        self._mode_label.disabled = True
        # case of a simplemodel is available for the user and the scheme
        if (self.user in self.state["simplemodel"]["existing"]) and (self.select_scheme.value in self.state["simplemodel"]["existing"][self.user]):
            self._mode_selection.options = ["deterministic","random","maxprob"]
            self._mode_label.disabled = False
            self._mode_label.options = self.state["schemes"]["available"][self.select_scheme.value]

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
        self.select_simplemodel.options = list(self.state["simplemodel"]["available"].keys())
        self.select_features.options = self.state["features"]["available"]
        if (self.user in self.state["simplemodel"]["existing"]) and (self.select_scheme.value in self.state["simplemodel"]["existing"][self.user]):
            current_model = self.state["simplemodel"]["existing"][self.user][self.select_scheme.value]["name"]
            self.simplemodel_params.value = json.dumps(self.state["simplemodel"]["existing"][self.user][self.select_scheme.value]["params"], 
                                                       indent=2)
            self.select_simplemodel.value = self.state["simplemodel"]["existing"][self.user][self.select_scheme.value]["name"]
        else:
            current_model = "No model available"
            #self.select_simplemodel.value = None
            self.simplemodel_params.value = ""
        self.simplemodel_state.value = f"Scheme : {self.select_scheme.value} - Current model: {current_model}"

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
        print(r)
        df = pd.DataFrame(r)
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
                "user":self.user
                }
        r = self._post("/schemes/add", 
                       params = params, 
                       json_data = data)
        print(r)
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
        data = {
                "model":model,
                "features":features,
                "params":json.loads(parameters),
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
        
        params = {"project_name":self.project_name}
        data = {
                "project_name":self.project_name,
                "scheme":self.select_scheme.value,
                "user":self.user,
                "name":f"_{self.user}", # générique
                "base_model":self.new_bert_base.value,
                "params":json.loads(self.new_bert_params.value),
                "test_size":0.2
                }
        
        r = self._post("/models/bert/train", 
                       params = params, 
                       json_data = data)
        print(r)
        time.sleep(2)
        self.bert_training = True
        self.update_tab_bertmodels()
        self.compute_new_bert.disabled = False
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
        print(r)
        time.sleep(2)
        self.bert_training = False
        self.compute_new_bert.disabled = False
        self.update_tab_bertmodels()
        return True

    def _get_previous_element(self) -> bool:
        """
        Load previous element in history
        """
        if len(self.history) < 1:
            print("No element in history")
            return False
        
        element_id = self.history.pop()
        r = self._get(route = f"/elements/{element_id}",
                      params = {"project_name":self.project_name})
        
        # Managing errors
        if "error" in r:
            print(r)
            return False

        # Update interface
        self.current_element = r
        self._textarea.value = r["text"]   

        return True     
    
    def compute_feature(self, feature_name) -> bool:
        """
        Compute feature
        """
        if not feature_name in self.state["features"]["options"]:
            return "This feature doesn't exist"
        r = self._post(f"/features/add/{feature_name}", 
                    params = {"project_name":self.project_name})
        print(r)
        self.update_tab_features()
        return True
    
    def delete_feature(self, feature_name) -> bool:
        """
        Delete existing feature
        """
        r = self._post(f"/features/delete/{feature_name}", 
                    params = {"project_name":self.project_name})
        print(r)
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
            "value":value
            }
        
        r = self._post("/features/add/regex",
            params = {"project_name":self.project_name},
            json_data=data)
        print(r)
        self.update_tab_features()
        return True
    
    def save_bert(self, name):
        params = {"project_name":self.project_name,
                  "user":self.user,
                  "name":name
                  }
        r = self._post("/models/bert/save",
            params = params)
        print(r)
        return r
        
    
    def periodic_update(self):

        while True:
            time.sleep(self.update_time)
            self.state = self.get_state()
            print(self.state["bertmodels"])
            # case of training finished
            if self.user in self.state["bertmodels"]["training"]:
                self.bert_training = True
            else:
                self.bert_training = False

                # return to normal state
                print("training finished")
                #self.bert_training = False
            self.update_tab_bertmodels(state=False)
    
    
    async def update_state(self):
        while True:
            self.state = self.get_state()
            await asyncio.sleep(self.update_time)

    def interface(self):
        """
        General interface
        - divided by tab
        """

        # updating thread
        asyncio.create_task(self.update_state())

        #------------
        # Tab schemes
        #------------
        self.select_scheme = widgets.Dropdown(description="Select: ", value="", options=[""])
        valid_delete_scheme = widgets.Button(description = "Delete", button_style = "danger")
        valid_delete_scheme.on_click(lambda b : self.delete_scheme(self.select_scheme.value))
        new_scheme = widgets.Text(description="New: ")
        valid_new_scheme = widgets.Button(description = "Create")
        valid_new_scheme.on_click(lambda b : self.create_scheme(new_scheme.value))
        self.select_label = widgets.Dropdown(description="Labels: ")
        valid_delete_label = widgets.Button(description = "Delete", button_style = "danger")
        valid_delete_label.on_click(lambda b : self.delete_label(self.select_label.value))
        new_label = widgets.Text(description="New label: ")
        valid_new_label = widgets.Button(description = "Create")
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
                            widgets.HBox([self.select_label, valid_delete_label]),
                            widgets.HBox([new_label, valid_new_label]),
                        ])

        #-----------
        # Tab codage
        #-----------
        self._textarea = widgets.Textarea(value="",
                                   layout=widgets.Layout(width='600px',height='150px'), 
                                   description='')
        self._back = widgets.Button(description = "back",layout=widgets.Layout(width='100px'))
        self._back.on_click(lambda x : self._get_previous_element())
        self._mode_selection = widgets.Dropdown(layout=widgets.Layout(width='120px'))
        self._mode_sample = widgets.Dropdown(layout=widgets.Layout(width='120px'))
        self._mode_label = widgets.Dropdown(layout=widgets.Layout(width='120px'),
                                            disabled=True)
        self._labels = widgets.HBox()

        # Populate
        self.update_tab_annotations()
        self._mode_selection.value = self._mode_selection.options[0]
        self._mode_sample.value = self._mode_sample.options[0]
        self._display_next()
        self._display_buttons_labels()

        # Group in tab
        tab_annotate = widgets.VBox([
                            #self._schemes,
                             widgets.HBox([self._back,
                                    self._mode_selection,
                                    self._mode_sample,
                                    self._mode_label]),
                              self._textarea,
                              self._labels
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
        self.data_description = widgets.Textarea(disabled=True, 
                                                 layout={'width': '400px', 'height':'300px'})

        # Populate
        self.update_tab_description()

        # Group in tab
        tab_description = widgets.VBox([self.data_description])

        #----------------
        # Tab SimpleModel
        #----------------
        self.simplemodel_state = widgets.Text(disabled=True)
        self.simplemodel_statistics= widgets.Text(disabled=True,
                                                  value = "Here put statistics")

        self.select_simplemodel =  widgets.Dropdown(description = "models")
        def on_change_scheme(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.simplemodel_params.value = json.dumps(self.state["simplemodel"]["available"][self.select_simplemodel.value],
                                                           indent=2)
        self.select_simplemodel.observe(on_change_scheme)
        self.select_features = widgets.SelectMultiple()
        self.simplemodel_params = widgets.Textarea(value="",
                                                   layout=widgets.Layout(width='300px',height='200px'))
        valid_model = widgets.Button(description = "⚙️Train")
        valid_model.on_click(lambda b : self.create_simplemodel(scheme=self.select_scheme.value, #attention il faudra revoir le choix du scheme
                                                               model = self.select_simplemodel.value,
                                                               parameters = self.simplemodel_params.value,
                                                               features = self.select_features.value))

        # Populate
        self.update_tab_simplemodel()

        # Group in tab
        tab_simplemodel = widgets.VBox([
                            widgets.HBox([self.simplemodel_state,self.simplemodel_statistics]),
                            self.select_simplemodel,
                             widgets.HBox([self.select_features,
                                    self.simplemodel_params]),
                              valid_model
             ])
        
        #-------------
        # Tab Features
        #-------------
        self.info_features  = widgets.HTML(value = "No process currently running")
        self.available_features =  widgets.Dropdown(description = "Available")
        delete_feature = widgets.Button(description = "Delete", button_style="danger")
        delete_feature.on_click(lambda x: self.delete_feature(self.available_features.value))
        self.add_features = widgets.Dropdown(description="Add: ", value="", options=[""])
        valid_compute_features = widgets.Button(description = "⚙️Compute")
        valid_compute_features.on_click(lambda x : self.compute_feature(self.add_features.value))
        valid_compute_features.style.button_color = 'lightgreen'
        add_regex_value = widgets.Text(description="Add regex:")
        #add_regex_name = widgets.Text(description="Name:")
        valid_regex = widgets.Button(description = "Add")
        valid_regex.style.button_color = 'lightgreen'
        valid_regex.on_click(lambda x: self.add_regex(add_regex_value.value))

        # Populate
        self.update_tab_features()

        # Group in tab
        tab_features = widgets.VBox([
            self.info_features,
            widgets.HBox([self.available_features,delete_feature]),
            widgets.HBox([self.add_features,valid_compute_features]),
            widgets.HBox([add_regex_value,valid_regex]),
             ])

        #--------------
        # Tab BertModel
        #--------------
        self.bert_status = widgets.Text(disabled=True)
        self.available_bert = widgets.Dropdown(description="Trained:")
        def on_change_model(change): # if select one, display its options on_select
            if change['type'] == 'change' and change['name'] == 'value':
                self.new_bert_params.value = "TO IMPLEMENT"
        self.available_bert.observe(on_change_model)

        self.new_bert_base = widgets.Dropdown(description="Base:")
        self.new_bert_params = widgets.Textarea(layout={'width': '200px','height':"200px"})
        self.compute_new_bert = widgets.VBox()

        self.bert_name = widgets.Text(description="Name:", layout={'width': '150px'}, value="Name")
        self.record_bert = widgets.Button(description = "Save Bert")
        self.record_bert.on_click(lambda x : self.save_bert(self.bert_name.value))

        # Populate
        self.update_tab_bertmodels()

        # Group in tab
        tab_bertmodel = widgets.VBox([
                                self.bert_status,
                                self.available_bert,
                                self.new_bert_base,
                                self.new_bert_params,
                                self.compute_new_bert,
                                widgets.HTML(value="<hr>"),
                                widgets.HBox([self.bert_name, self.record_bert])
                             ])


        # display global widget
        self.output = widgets.Tab([tab_schemes,
                                   tab_annotate,
                                   tab_description,
                                   tab_data,
                                   tab_features,
                                   tab_simplemodel,
                                   tab_bertmodel],
                                  titles = ["Schemes",
                                            "Annotate",
                                            "Description",
                                            "Data",
                                            "Features",
                                            "SimpleModels",
                                            "BertModels"])
        
        # Update everything on tab change
        def on_tab_selected(change):
            self.update_global()
        self.output.observe(on_tab_selected, names='selected_index')

        # Display
        clear_output()
        display(self.output)