import ipywidgets as widgets
from IPython.display import display
import json

from project import Schemes
from server import Server

import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# Est-ce qu'il ne faudrait pas faire passer la liste des modes
# de sélection en dynamique (envoyé par le backend)
# Ca éviterait de sélectionner a priori l'étiquette

# gérer la forme des requêtes
# {"project_name","type","content" + "user"}

# gérer l'interface dynamique : 
# - créer un nouveau système de codage

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
        self.schemes: None|Schemes= None
        self.simplemodel_params: dict = {}
        self.state = None # all current parameters #TODO be sure to refresh
        self.next = {
                    "mode":"deterministic",
                    "on":"untagged",
                    "label":None
                    }
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
        self.state = self.get_next()
        self.simplemodel_params = self.get_simplemodel_params() # TODO : vérifier l'actualisation
        self.create_widget()
        logging.info("Connecting and launching widget")
        
    def get_schemes(self):
        s = Schemes(self.project_name)
        req = {
                "project_name":self.project_name,
                "type":"schemes"
            }
        req = self.server.get(req)
        s.load(req["content"])
        return s
    
    def get_simplemodel_params(self):
        req = {
                "project_name":self.project_name,
                "type":"simplemodel"            
        }
        req = self.server.get(req)
        return req["content"]
    
    def get_next(self):
        """
        Get next element to annotate
        """
        req = {
                "project_name":self.project_name,
                "type":"next",
                "content":self.next
              }
        req = self.server.get(req)
        logging.info(f"Get next element from server {req}")
        if "error" in req:
            print(req)
            raise ConnexionError
        return req
    
    def get_element(self,element_id):
        """
        Get element by id
        """
        req = {
                "project_name":self.project_name,
                "type":"element",
                "element_id":element_id
              }
        logging.info(f"Get a specific element from server {req}")
        return self.server.get(req)

    def update_schemes(self):
        """
        Update scheme
        """
        req = {
                "project_name":self.project_name,
                "type":"update_schemes",
                "content":self.schemes.dump()
            }
        self.server.post(req)
        #self.params = self.get_params() #update params
        return {"send":"ok"}

    def push_label(self,element_id,label):
        """
        Push annotation
        """
        req = {
                "project_name":self.project_name,
                "type":"label",
                "content":  {
                            "element_id":element_id,
                            "label":label
                            }
            }
        req = self.server.post(req)

    def delete_label(self,element_id):
        """
        Delete annotation
        """
        req = {
                "project_name":self.project_name,
                "type":"delete_label",
                "content":{"element_id":element_id}

        }
        req = self.server.post(req)

    def __on_button_click(self,b):
        """
        Validation of a label
        """
        # push label, get next element, change text
        self.push_label(self.state,b.description)
        self.history.append(self.state)
        self.state = self.get_next()
        self.components["text"].value = self.state["content"]["text"]
        
    def __back(self,b):
        """
        Reverse operation
        """
        self.delete_label(self.state["element_id"])
        self.state = self.history.pop()
        self.components["text"].value = self.state["content"]["text"]
        return None
    
    def __add_label_to_scheme(self,a):
        """
        Add a category, modify widget and save params
        """
        # test if not exist
        if a.value in self.schemes.labels:
            return None
        # change schemes
        self.schemes.labels.append(a.value)
        self.update_schemes()
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
        self.update_schemes() # update server
        self.components["current_scheme"].children = [i for i in self.components["current_scheme"].children if i.description != b.description]
        self.components["delete_labels"].children = [i for i in self.components["delete_labels"].children if i.description != b.description]
        b.close() # destroy button

    def push_simplemodel(self):
        req = {
                "type":"simplemodel",
                "project_name":self.project_name,
                "content": {
                        "current": self.components["models"].children[0].value,
                        "features":self.components["models"].children[1].value,
                        "parameters":json.loads(self.components["models"].children[2].value)}
                }
        # TODO: tester la formation de la chaine JSON
        req = self.server.post(req)
        if "error" in req:
            self.__display_info(str(req))
        else:
            # update widget with new menu
            self.components["header"].children[1].options += ("maxprob",)
            

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
            disabled=True,
        )
        def on_change_label(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.next["label"] = change['new'].lower()
        mode_label.observe(on_change_label)

        mode_menu = widgets.Dropdown(
            options=self.state["options"]["selection_mode"],
            value='deterministic',
            description='Selection :',
            layout={'width': '200px'},
            disabled=False,
        )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.next["mode"] = change['new'].lower()
            # Case of maxprob
            if self.next["mode"].lower() == "maxprob":
                mode_label.disabled = False
                self.next["label"] = self.schemes.labels[0]
            if self.next["mode"].lower() != "maxprob":
                mode_label.disabled = True
                self.next["label"] = None
            
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
                self.next["on"] = change['new'].lower()
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
        self.components["text"] = widgets.Textarea(value=self.state["content"]["text"],
                                        layout=widgets.Layout(width='700px',height='150px'), 
                                        description='')
        # Annotation
        
        # A button for each category
        self.components["current_scheme"] = widgets.HBox([self.__add_button_label(i) for i in self.schemes.labels])

        # Menu to select scheme
        schemes = widgets.Select(
                    options= self.schemes.available.keys(),
                    description='',
                    disabled=False,
                    layout={'width': '200px',"height":"50px"}
                    )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                print("Pas encore implémenté ",change)
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

        # Simple model parameters
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
        
        select_embeddings = widgets.SelectMultiple(
            options=self.state["options"]["available_features"],
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
                                                  select_embeddings,
                                                  model_params,
                                                  model_valid])

        # Tabs for the two menus
        self.components["footer"] = widgets.Tab([
                        self.components["current_scheme"],
                        widgets.HBox([
                                    schemes,
                                    widgets.VBox([new_cat,self.components["delete_labels"]])]
                                    ),
                        widgets.Text(disabled = True,value="Embeddings"),
                        self.components["models"]

                        ], 
                        titles = ['Annotations', 
                          'Modifications', 'Embeddings', 'Models'])
        
        # Display the widget
        display(self.components["header"])
        display(self.components["text"])
        display(self.components["footer"])