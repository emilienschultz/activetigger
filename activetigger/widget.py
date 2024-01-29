import ipywidgets as widgets
from IPython.display import display
import json
import re
import time

from project import Schemes
from server import Server

import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# gérer la forme des requêtes
# {"project_name","type","content" + "user"}

# Unifier la gestion des informations de state ? 
# intéger par exemple simplemodel
# ajouter un décorateur

# TODO : SELECTIONNER UN NOUVEAU SCHEME

class ConnexionError(Exception):
    def __init__(self, message="Error during the connexion"):
        self.message = message
        super().__init__(self.message)

class Widget():
    """
    Widget
    """
    def __init__(self) -> None:        
        self.server: None|Server = None # endpoint
        self.project_name: None|str = None
        self.state = None # global parameters for this instance
        self.current = None # current element to tag
        self.schemes: None|Schemes= None # managing schemes
        self.bert: None|dict = None # managing BERT params
        self.simplemodel_params: dict = {} # managing models
        self.history: list = []
        self.components: dict = {}
        logging.info("Start widget")
    
    def connect(self, 
                server: Server,
                project_name: str) -> None:
        """
        Initialize the frontend
        """
        self.server = server
        self.project_name = project_name
        self.schemes = self.get_schemes()
        self.state = self.get_state()
        self.current = self.get_next()
        self.simplemodel_params = self.get_simplemodel_params() # TODO : vérifier l'actualisation 
        self.bert = self.get_bert()          
        self.create_widget()
        logging.info("Connecting and launching widget")
        

    def get_schemes(self) -> Schemes:
        """
        Getting available schemes
        """
        s = Schemes(self.project_name)
        req = {
                "project_name":self.project_name,
                "type":"schemes"
            }
        rep = self.server.get(req)
        s.load(rep["content"])
        return s
    
    def get_simplemodel_params(self):
        """
        Getting available simplemodels
        """
        req = {
                "project_name":self.project_name,
                "type":"simplemodel"            
            }
        rep = self.server.get(req)
        return rep["content"]
    
    def get_bert(self):
        """
        Getting bert params
        """
        req = {
                "project_name":self.project_name,
                "type":"bert"            
        }
        rep = self.server.get(req)
        return rep["content"]
    
    def get_state(self):
        """
        Get state of the project in the server
        """
        req = {
                "project_name":self.project_name,
                "type":"state"
                }
        rep = self.server.get(req)
        return rep["content"]

    def get_next(self) -> dict:
        """
        Get next element to annotate
        """
        # update current scheme (to move somewhere else)
        state = self.state
        state["scheme"]["current"] = self.schemes.name
        req = {
                "project_name":self.project_name,
                "type":"next",
                "content":state
              }
        rep = self.server.get(req) 
        if "error" in rep:
            raise ConnexionError
        return rep
    
    def get_element(self,element_id) -> dict:
        """
        Get element by id
        """
        req = {
                "project_name":self.project_name,
                "type":"element",
                "content":{"element_id":element_id,
                           "scheme":self.schemes.name}
              }
        rep = self.server.get(req)
        return rep

    def post_schemes(self) -> dict:
        """
        Update scheme
        """
        req = {
                "project_name":self.project_name,
                "type":"update_schemes",
                "content":self.schemes.dump()
            }
        self.server.post(req)
        return {"send":"ok"}

    def post_tag(self, element_id:int|str, label:str) -> dict:
        """
        Push annotation
        """
        req = {
                "project_name":self.project_name,
                "type":"label",
                "content":  {
                            "scheme":self.schemes.name,
                            "element_id":element_id,
                            "label":label
                            }
            }
        rep = self.server.post(req)
        return rep

    def delete_tag(self, element_id:int|str) -> dict:
        """
        Delete annotation
        """
        req = {
                "project_name":self.project_name,
                "type":"delete_label",
                "content":{"element_id":element_id}

        }
        rep = self.server.post(req)
        return rep

    def __on_button_click(self,b) -> None:
        """
        Validation of a label
        """
        # push label, get next element, change text
        self.post_tag(self.current["element_id"],b.description)
        self.history.append(self.current)
        self.current = self.get_next()
        self.components["text"].value = self.current["content"]["text"]
        return None

        
    def __back(self,b) -> None:
        """
        Reverse operation
        """
        self.delete_tag(self.current["element_id"])
        self.current = self.history.pop()
        self.components["text"].value = self.current["content"]["text"]
        return None
    
    def __add_label_to_scheme(self,a) -> None:
        """
        Add a category, modify widget and save params
        """
        # test if not exist
        if a.value in self.schemes.labels:
            return None
        # change schemes
        self.schemes.labels.append(a.value)
        self.post_schemes()
        # change widget
        self.components["current_scheme"].children += (self.__add_button_label(a.value),)
        self.components["delete_labels"].children += (self.__add_button_remove_label(a.value),)
        return None

    def __add_button_label(self,c):
        """
        Add a button label
        """
        b = widgets.Button(description=c)
        b.on_click(self.__on_button_click)
        return b

    def __display_info(self,info):
        """
        Display info in the widget
        """
        self.information.value = info

    def __add_button_remove_label(self, c):
        """
        Add a button remove label
        """
        b = widgets.Button(description=c)
        b.on_click(self.__remove_label_from_scheme)
        return b
    
    def __remove_label_from_scheme(self,b):
        self.schemes.labels.remove(b.description) #remove the label
        self.post_schemes() # update server
        self.components["current_scheme"].children = [i for i in self.components["current_scheme"].children if i.description != b.description]
        self.components["delete_labels"].children = [i for i in self.components["delete_labels"].children if i.description != b.description]
        b.close() # destroy button

    def push_simplemodel(self):
        req = {
                "type":"simplemodel",
                "project_name":self.project_name,
                "content": {
                        "current": self.components["models"].children[0].value,
                        "features":self.components["models"].children[2].value,
                        "parameters":json.loads(self.components["models"].children[2].value)}
                }
        # TODO: tester la formation de la chaine JSON
        req = self.server.post(req)
        if "error" in req:
            self.__display_info(str(req))
        else:
            # update widget with new menu
            self.components["header"].children[1].options += ("maxprob",)

    def is_valid_regex(self, pattern):
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False

    def delete_feature(self):
        """
        Delete available feature
        """
        feature = self.components["features"].children[0].value
        req = {
            "project_name":self.project_name,
            "type":"delete_feature",
            "content":{"name":feature}
        }
        rep = self.server.post(req)

        # update
        self.components["features"].children[0].options = self.state["features"]["available_features"]
        self.components["models"].children[2].options = self.state["features"]["available_features"]

        return None

    def post_regex(self):
        """
        Add new regex feature
        """
        regex = self.components["features"].children[2].value
        if self.is_valid_regex(regex):
            name = "regex_"+regex
            req = {
                "project_name":self.project_name,
                "type":"regex",
                "content":{"name":name,
                           "value":regex}
            }
            rep = self.server.post(req)
            # update widget : state, available features, simplemodel
            # open question : why available_features update ?
            self.components["features"].children[0].options = self.state["features"]["available_features"]
            self.components["models"].children[2].options = self.state["features"]["available_features"]

    def create_scheme(self, b, name:str):
        # push the scheme to server
        req = {
            "project_name":self.project_name,
            "type":"new_scheme",
            "content":{"name":name}
        }
        rep = self.server.post(req)
        # update the interface
        #self.schemes.add(name, []) # TODO : POURQUOI IL Y A UNE SYNCRO SERVEUR/FRONT ACTUELLEMENT ??
        self.components["modify_schemes"].children[0].options+=(name,)
        return None

    def train_bert(self):
        """
        Train Bert Model
        """

        req = {
            "project_name":self.project_name,
            "type":"train_bert",
            "content":{
                "name":self.components["bert"].children[1].value,
                "type":self.components["bert"].children[2].value,
                "params":json.loads(self.components["bert"].children[3].value)
            }
        }

        rep = self.server.post(req)
        return None

    def create_widget(self):
        """
        Create the widget
        """
        
        # Back button
        retour = widgets.Button(description="Back",
                                icon="backward")
        retour.on_click(self.__back)

        # Select mode

        mode_label = widgets.Dropdown(
            options=self.schemes.labels,
            description='',
            value = self.schemes.labels[0],
            layout={'width': '100px'},
            disabled=True)
        def on_change_label(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.state["mode"]["label"] = change['new'].lower()
        mode_label.observe(on_change_label)

        mode_menu = widgets.Dropdown(
            options=self.state["mode"]["available_modes"],
            value='deterministic',
            description='Selection :',
            layout={'width': '200px'},
            disabled=False,
        )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.state["mode"]["mode"] = change['new'].lower()
            # Case of maxprob
                if self.state["mode"]["mode"].lower() == "maxprob":
                    mode_label.disabled = False
                    self.state["mode"]["label"] = self.schemes.labels[0]
                if self.state["mode"]["mode"].lower() != "maxprob":
                    mode_label.disabled = True
                    self.state["mode"]["label"] = None
            
        mode_menu.observe(on_change)

        mode_rows = widgets.Dropdown(
            options=["All","Tagged","Untagged"],
            value='Untagged',
            description='',
            layout={'width': '100px'},
            disabled=False,
        )
        def on_change_mode_rows(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.state["mode"]["on"] = change['new'].lower()
        mode_rows.observe(on_change_mode_rows)

        # Information
        self.information = widgets.Text(disabled=True,layout={'width': '200px'})
        self.__display_info('Widget initialized')

        self.components["header"] = widgets.HBox([retour,
                                                  mode_menu,
                                                  mode_rows,
                                                  mode_label,
                                                  self.information])
        
        # Text area
        self.components["text"] = widgets.Textarea(value=self.current["content"]["text"],
                                        layout=widgets.Layout(width='700px',height='150px'), 
                                        description='')
        # Annotation
        
        # A button for each category
        self.components["current_scheme"] = widgets.HBox([self.__add_button_label(i) for i in self.schemes.labels])

        #----------
        # Modify schemes
        #----------

        # Menu to select scheme
        schemes = widgets.Select(
                    options= self.schemes.available.keys(),
                    description='',
                    disabled=False,
                    layout={'width': '200px',"height":"50px"}
                    )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # use the new scheme & display the buttons
                self.schemes.select(change["new"])
                self.components["current_scheme"].children = [self.__add_button_label(i) for i in self.schemes.labels]
                self.components["delete_labels"].children = [self.__add_button_remove_label(i) for i in self.schemes.labels]
                print("changement de schemes")
        schemes.observe(on_change)

        # Field to add a category
        new_cat = widgets.Text(
            value='Label',
            placeholder='Type something',
            disabled=False,
            description="Add",
            layout={'width': '200px'}   
            )
        new_cat.on_submit(self.__add_label_to_scheme)    
        
        self.components["delete_labels"] = widgets.VBox([self.__add_button_remove_label(i) for i in self.schemes.labels])

        # new scheme
        new_scheme_name = widgets.Text(
            description="New scheme",
            value="Name",
            layout={'width': '200px'}
        )
        new_scheme_button = widgets.Button(description="Add scheme")
        new_scheme_button.on_click(lambda b : self.create_scheme(b, new_scheme_name.value))

        self.components["modify_schemes"] = widgets.HBox([
                                    schemes,
                                    widgets.VBox([new_cat,self.components["delete_labels"]]),
                                    widgets.VBox([new_scheme_name,new_scheme_button])
                                    ]
                                    )
        #----------
        # SimpleModel tab
        #----------
        
        # TODO : il manque la sélection des embeddings + du label d'intérêt

        select_model =  widgets.Dropdown(
                    options=self.simplemodel_params["available"].keys(),
                    description='',
                    value=None
                )
        def on_change_model(change):
            if change['type'] == 'change' and change['name'] == 'value':
                # display parameters of the selected model
                model_params.value = json.dumps(self.simplemodel_params["available"][change['new']])
        select_model.observe(on_change_model)
        
        select_features = widgets.SelectMultiple(
            options=self.state["features"]["available_features"],
            value=[], 
            description='Features'
        )

        model_params = widgets.Textarea(
            layout={'width': '200px'},
            value=json.dumps(self.simplemodel_params["current"]))
        
        # TODO : afficher paramètre par défaut des modèles on-change "available"
        # ou les paramètres du modèle actuel "current"
        # et gérer l'envoi des bons paramètres pour l'entrainement du modèle
        
        model_valid = widgets.Button(description="⚙️ Compute")
        def update_simplemodel(b):
            self.push_simplemodel()
        model_valid.on_click(update_simplemodel)
        self.components["models"] = widgets.HBox([select_model,
                                                  select_features,
                                                  model_params,
                                                  model_valid])
        
        #-------------
        # Features tab
        #-------------

        available_features = widgets.Dropdown(
                            options=self.state["features"]["available_features"],
                            value=self.state["features"]["available_features"][0],
                            layout={'width': '150px'}
                        )
        delete_feature = widgets.Button(description="Delete feature",
                                        layout={'width': '150px'})
        def func_delete_feature(b):
            self.delete_feature()
            b.disabled = True
            time.sleep(1) 
            b.disabled = False
        delete_feature.on_click(func_delete_feature)

        enter_regex = widgets.Textarea(layout={'width': '200px'},
            value="Regex predictors")
        valid_regex = widgets.Button(description="Add regex")
        def add_regex(b):
            self.post_regex()
            b.disabled = True
            time.sleep(1) 
            b.disabled = False
        valid_regex.on_click(add_regex)

        self.components["features"] = widgets.HBox([available_features,delete_feature,
                                                    enter_regex,valid_regex])


        #----------
        # BERT tab
        #----------
        # bouton save ?

        bert_available = widgets.Dropdown(
            options=self.bert["models"]["trained"],
            layout={'width': '200px'},
            disabled=False)

        bert_name = widgets.Text(value="Name",layout={'width': '150px'})

        bert_choice = widgets.Dropdown(
            options=self.bert["models"]["available"],
            layout={'width': '200px'},
            disabled=False)
        #TODO : selection du modèle
        
        bert_params = widgets.Textarea(
            layout={'width': '150px',"height":"200px"},
            value=json.dumps(self.bert["params"]["default"]))
        
        bert_train = widgets.Button(description="⚙️ Train")
        model_valid.on_click(self.train_bert)

        self.components["bert"] = widgets.HBox([bert_available,
                                                bert_name, 
                                                bert_choice,
                                                bert_params, 
                                                bert_train])

        # ------------
        # Tabs
        # ------------
        self.components["footer"] = widgets.Tab([
                        self.components["current_scheme"],
                        self.components["modify_schemes"],
                        self.components["features"],
                        self.components["models"],
                        self.components["bert"],
                        widgets.Textarea(value="Statistics",disabled=True)
                        ], 
                        titles = ['Annotations', 
                          'Modifications', 'Features',
                          'Simple Models','Bert Models',
                           'Statistics'])
        
        # Display the widget
        display(self.components["header"])
        display(self.components["text"])
        display(self.components["footer"])