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
    def __init__(self):        
        self.server: None|Server = None # endpoint
        self.project_name: None|str = None
        self.schemes: None|Schemes= None
        self.simplemodel_params: dict = {}
        self.state = None
        self.history: list = []
        self.components: dict = {}
        logging.info("Start widget")
    
    def connect(self,server,project_name):
        """
        Initialize the frontend
        """
        self.project_name = project_name
        self.server = server
        self.mode = "deterministic"
        self.schemes = self.get_schemes()
        self.state = self.get_next()
        self.simplemodel_params = self.get_simplemodel_params()
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
    
    def get_next(self,mode="deterministic"):
        """
        Get next element to annotate
        """
        req = {
                "project_name":self.project_name,
                "type":"next",
                "mode":mode
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
        self.state = self.get_next(self.mode)
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
                "type":"update_simplemodel",
                "project_name":self.project_name,
                "content": {
                        "current": self.components["models"].children[0].value,
                        "embeddings":self.components["models"].children[1].value,
                        "parameters":self.components["models"].children[2].value}
                }
        req = self.server.post(req)

    def create_widget(self):
        """
        Create the widget
        """
        
        # Back button
        retour = widgets.Button(description="Back",
                                icon="backward")
        retour.on_click(self.__back)

        # Select mode
        mode_menu = widgets.Dropdown(
            options=['Deterministic', 'Random','MaxProb'],
            value='Deterministic',
            description='Selection :',
            disabled=False,
        )
        
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                self.mode = change['new'].lower()

        mode_menu.observe(on_change)

        # Information
        self.information = widgets.Text(disabled=True,layout={'width': '200px'})
        self.__display_info('Widget initialized')

        self.components["header"] = widgets.HBox([retour,mode_menu,self.information])
        
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
                    options=self.simplemodel_params["available"],
                    description=''
                )
        
        options_list = ["embedding 1","embedding 2"]
        select_embeddings = widgets.SelectMultiple(
            options=options_list,
            value=[], 
            description='Embeddings'
        )

        model_params = widgets.Textarea(
            value=json.dumps(self.simplemodel_params["parameters"]))
        
        model_valid = widgets.Button(description="⚙️ Compute")
        def change_simplemodel(b):
            self.push_simplemodel()
        model_valid.on_click(change_simplemodel)
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