import ipywidgets as widgets
from IPython.display import display, clear_output
import json
import requests as rq
from pathlib import Path
import pandas as pd
import io
import re
import time

URL_SERVER = "http://127.0.0.1:8000/"
headers = {'x-token': 'your_token'}


class ConnexionError(Exception):
    def __init__(self, message="Error during the connexion"):
        self.message = message
        super().__init__(self.message)

class Widget():
    """
    Widget
    """
    def __init__(self) -> None:
        self.user = "local"
        self.project_name: None|str = None
        self.scheme: dict = {
                            "current":None,
                            "mode":None,
                            "on":None,
                            "tag":None,
                            }
        self.screen = None

    def _post(self,
             route:str, 
             params:dict|None = None, 
             files:str|None = None,
             data:dict|None = None):
        url = URL_SERVER + route
        r = rq.post(url, 
                    params = params,
                    data = data,
                    files=files,
                    headers=headers)
        return json.loads(r.content)
    
    def _get(self,
             route:str, 
             params:dict|None = None, 
             data:dict|None = None):
        url = URL_SERVER + route
        r = rq.get(url, 
                    params = params,
                    data = data,
                    headers=headers)
        print(r.content)
        return json.loads(r.content)

    def start(self):
        """
        Menu to start the widget
        - connect existing project
        - start a new one
        Add -> delete ?
        """
        # Get existing projects
        existing = self._get("projects")

        # Existing projects
        existing_projects = widgets.Dropdown(
            options=existing["existing projects"],
            description='Available :',
            #value = "",
            layout={'width': '300px'},
            disabled=False)

        # Start existing project
        start = widgets.Button(description="Connecter")
        def start_project(b):
            self.project_name = existing_projects.value
            self.state = self.get_state()
            self.interface()
        start.on_click(start_project)

        # Create a new project
        create = widgets.Button(description="Nouveau projet")
        def create_project(b):
            self._create_new_project()
        create.on_click(create_project)

        # Display
        clear_output()
        self.output = widgets.HBox([existing_projects, start, create])
        display(self.output)

    def get_state(self):
        state = self._get(route = f"state/{self.project_name}")
        return state

    def _create_new_project(self):
        """
        Create a new project
        """
        clear_output()
        # project name
        project_name = widgets.Text(disabled=False,
                                    description="Name:",
                                    layout={'width': '200px'})
        
        # select columns
        column_text = widgets.Dropdown(
            options=[],
            description='Text:',
            disabled=False)

        column_id = widgets.Dropdown(
            options=[],
            description='Id:',
            disabled=False)

        # load file
        file = widgets.Text(disabled=False,
                            description="Path:",
                            layout={'width': '200px'})
        load = widgets.Button(description="Load",
                              layout={'width': '100px'})
        def load_file(b):
            df = self._load_file(file.value)
            column_text.options = df.columns
            column_id.options = df.columns
        load.on_click(load_file)
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

        validate = widgets.Button(description="Create",
                              layout={'width': '100px'})
        def create_project(b):
            data = {
                    "project_name": project_name.value,
                    "col_text": column_text.value,
                    "col_id":column_id.value,
                    }
            files = {'file': (file.value,
                              open(file.value, 'rb'))}
            self._post(route="projects/new", 
                       data=data,
                       files=files)
            print("créer le projet")
            self.start()
        validate.on_click(create_project)

        self.output = widgets.VBox([project_name, 
                                    widgets.HBox([file, load]),
                                    widgets.HBox([column_text, column_id]),
                                    validate
                                    ])
        display(self.output)

    def _load_file(self,path):
        """
        Load file
        """
        path = Path(path)
        if not path.exists():
            return "File doesn't exist"
        if not path.suffix == '.csv':
            return "File not csv"
        df = pd.read_csv(path)
        return df
    
    def _send_tag(self,tag):
        # envoyer le tag
        # récupérer un nouveau text à coder 
        # actualiser
        return None


    def interface(self):
        #-----------
        # Tab codage
        #-----------
        self._textarea = widgets.Textarea(value="",
                                   layout=widgets.Layout(width='400px',height='150px'), 
                                   description='')
        self._schemes = widgets.Dropdown()
        self._back = widgets.Button()
        self._mode_selection = widgets.Dropdown()
        self._mode_type = widgets.Dropdown()
        self._mode_label = widgets.Dropdown()
        self._labels = widgets.HBox()

        # populate
        self._schemes.options = list(self.state["schemes"]["available"].keys())
        self._schemes.value = self._schemes.options[0]

        self._mode_type.options = self.state["next"]["sample"]
        self._mode_type.value = self._mode_type.options[0]

        self._mode_selection.options = self.state["next"]["methods"]
        self._mode_selection.value = self._mode_selection.options[0]

        # group in tab
        tab_annotate = widgets.VBox([
                            self._schemes,
                             widgets.HBox([self._back,
                                self._mode_selection,
                                self._mode_type,
                                self._mode_label]),
                              self._textarea,
                              self._labels
             ])

        self.output = widgets.Tab([tab_annotate,tab_annotate],
                                  titles = ["Annotate","Test"])
        def on_tab_selected(change):
            selected_tab_index = change['new']
            print(f"Tab {selected_tab_index + 1} selected")
        self.output.observe(on_tab_selected, names='selected_index')


        # Afficher
        clear_output()
        display(self.output)
