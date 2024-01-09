import os
from pathlib import Path
import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import functools
import numpy as np
import yaml

# scripts activetigger
import embeddings as emb

__version__ = "0.0.2"

# TO DO
# - repenser l'organisation des boutons
# - calculer les embeddings (boutons spécifique ?)
# - faire les prédictions
# - implémenter la suppression de label
# - passer tous les paramètres dans un json plutôt que des valeurs séparées
# - problème de l'affichage (concurrence entre affichage & effaçage)
# - déplacer les fonctions d"'accès aux données à la classe database
# - faut-il une classe modèle à part entière ?
# - ajouter des tests
# - réfléchir l'ergonomie du widget


# - implémenter la gestion de projets (fichier YAML avec les paramètres + répertoire)
# nom de projet : si existe déjà

class Database():
    """
    Data management

    Pour le moment choix de rester dans une gestion physique dossier/fichiers
    Dans le futur -> passer en base de données
    """

    def __init__(self, project_name : str, file: str, n_rows: int = 0):
        """
        Init the database
        
        For the moment in a csv file
        Add two new columns : 
        - codage (label)
        - proba (maxprob)
        """

        self.file = file #storage of the data
        self.project_name = project_name
        self.sbert = False
        self.fasttext = False
        self.content = pd.read_csv(file,index_col=0,
                                   low_memory=False,
                                   nrows=n_rows)
        self.n_rows = n_rows
        if n_rows == 0 :
            n_rows = len(self.content)
            
        # add features
        if not "label" in self.content.columns:
            self.content["label"] = None
        if not "proba" in self.content.columns:
            self.content["proba"] = None
            self.__predict_simple_model()

    def save(self):
        """
        Save the data
        """
        self.content.to_csv(self.file)
    
    def __predict_simple_model(self,model=None):
        """
        Fonction temporaire qui génère une prédiction
        """
        self.content["prob"] = np.random.rand(len(self.content))

    def compute_embeddings(self,fasttext=False, sbert=False):
        """
        Compute embeddings for simple models
        """
        if fasttext:
            None
        if sbert:
            None
        
        return None

                       
class Annotator():
    """
    Widget
    """
    
    def __init__(self, project_name, file: str = None, col_text: str = "test", cat: list = [], n_rows: int = 1000):
        """
        Parameters
        ----------
        project_name (str): project name
        file (str): path for the original data in CSV
        col_text (str): column containing the text
        cat (list): list of labels to use
        n_row (int) : number of rows of the dataset to use (especially for embeddings)
        """

        # Retrieve existing project or create a new : projects are directory with a name
        if Path(project_name).exists(): 
            #cas où le projet existe déjà
            print(f"Le projet {project_name} existe. Chargement des données")
            self.project_name = project_name
            params = self.__load_params()
            self.file = params["file"]
            self.n_rows = params["n_rows"]
            self.col_text = params["col_text"]
            self.cat = params["cat"]
        elif file is None: 
            #cas où le projet n'existe pas mais les données ne sont pas indiquées
            print("Le projet n'existe pas, veuillez indiquer un fichier de données à charger pour le créer")
        else: 
            # cas de création d'un projet
            print("Création d'un nouveau projet")
            os.makedirs(project_name)
            self.project_name = project_name
            self.file = file
            self.n_rows = n_rows
            self.col_text = col_text
            self.cat = cat
            self.__save_params()
            # préparer les données à partir du fichier originel (améliorer cette partie ?)
            self.corpus = Database(self.project_name, self.file,n_rows=self.n_rows)
            self.corpus.file = f"{self.project_name}/{self.project_name}.csv"
            self.corpus.save()
            self.corpus = None
        
        # Start labelling
        self.corpus = Database(self.project_name,f"{self.project_name}/{self.project_name}.csv", self.n_rows) 
        self.coded_items = []
        self.line = None
        self.select_mode = "Deterministic"
        self.line = self.__get_element()
        self.create_widget()
        return None
    
    def __repr__(self):
        return f"""
Annotation de corpus du fichier {self.file}
Avec les catégories {self.cat}
Actuellement il y a {len(self.corpus.content["label"].dropna())} éléments codés sur {len(self.corpus.content)}
"""

    def __load_params(self):
        """
        Load YAML configuration file
        """
        with open(f"{self.project_name}/{self.project_name}.yaml") as f:
            return yaml.safe_load(f)

    def __save_params(self):
        """
        Save YAML configuration file
        """
        params = {
                  "project_name":self.project_name,
                  "file":self.file,
                  "n_rows":self.n_rows,
                  "col_text":self.col_text,
                  "cat":self.cat
                 }
        with open(f"{self.project_name}/{self.project_name}.yaml", 'w') as f:
            yaml.dump(params, f)

    def __get_element(self,strategy="Deterministic"):
        """
        Get the element to annotate
        """
        # gérer le fait que tous les éléments sont codés ? / message d'alerte
        
        f = pd.isnull(self.corpus.content["label"])
        strategy = self.select_mode
        
        if strategy == "Deterministic": #first unannotated element
            return self.corpus.content[f].index[0]
        if strategy == "Random": #random element
            return self.corpus.content[f].sample(random_state=42).index[0]
        if strategy == "MaxProb": # max probability
            v = self.corpus.content[f].sort_values("prob",ascending=False).index[0]
            self.information.clear_output()
            print(v)
            #self.information.append_stdout("Test")
            return v
            
        # implémenter ici des stratégies plus complexes basées sur du active learning
    
    def __get_text(self):
        """
        Get the content to display
        """
        return self.corpus.content.loc[self.line,self.col_text]
    
    def __add_annotation(self,code):
        """
        Add annotation
        """
        self.corpus.content.loc[self.line,"label"] = code
        self.coded_items.append(self.line)
        self.line = self.__get_element()

    def __on_button_click(self,b):
        """
        Action if button clicked
        """
        # push the current code and change the current line
        self.__add_annotation(b.description)
        # change the text displayed
        self.widget_text.value = self.__get_text()
        
    def __back(self,b):
        """
        Reverse operation
        """
        self.line = self.coded_items.pop()
        self.corpus.content.loc[self.line,"label"] = None
        self.widget_text.value = self.__get_text()
        return None
    
    def __add_cat(self,a):
        """
        Add a category to cat
        """
        self.cat.append(a.value)
        self.boutons.children += (self.__add_button(a.value),)
        self.save_params() #update yaml file
        return None
        
    def __add_button(self,c):
        """
        Add a button un the HBOX
        """
        b = widgets.Button(description=c)
        b.on_click(self.__on_button_click)
        return b

    def __display_info(self,info):
        #self.information.clear_output()
        #self.information.append_stdout(info)
        self.information.value = info
    
    def create_widget(self):
        """
        Create the widget
        """

        # Header
        
        # Back button
        retour = widgets.Button(description="Revenir en arrière",
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
                self.select_mode = change['new']

        mode_menu.observe(on_change)


        # Information
        self.information = widgets.Text(disabled=True,layout={'width': '200px'})
        self.__display_info('Widget initialized')

        self.widget_header = widgets.HBox([retour,mode_menu,self.information])
        
        # Text area
        self.widget_text = widgets.Textarea(value=self.__get_text(),
                                        layout=widgets.Layout(width='700px',height='150px'), 
                                        description='')

        # Annotation
        
        # Field to add a category
        new_cat = widgets.Text(
            value='Add',
            placeholder='Type something',
            disabled=False   
            )
        new_cat.on_submit(self.__add_cat)    

        # Field to delete a category
        delete_cat = widgets.Select(
        options=["None"]+self.cat,
        description='Delete:',
        disabled=False
        )
        def on_change(change):
            if change['type'] == 'change' and change['name'] == 'value':
                print("Pas encore implémenté ",change)
        delete_cat.observe(on_change)



        
        # A button for each category
        self.boutons = widgets.HBox([self.__add_button(i) for i in self.cat])

        # Tabs for the two menus
        self.widget_footer = widgets.Tab()
        self.widget_footer.children = [self.boutons,
                        widgets.HBox([new_cat,delete_cat])]
        self.widget_footer.titles = ['Annotations', 'Modifications']
        
        # Display the widget
        display(self.widget_header)
        display(self.widget_text)
        display(self.widget_footer)