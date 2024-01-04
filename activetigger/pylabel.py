import pandas as pd
import ipywidgets as widgets
from IPython.display import display
import functools

__version__ = "0.0.2"

# TO DO
# ajouter des tests
# réfléchir l'ergonomie du widget


class Labeller():
    """
    Widget to label a corpus
    
    """
    
    def __init__(self,file: str,col_text: str,cat: list):  
        """
        Parameters
        ----------
        file (str): path for the sample to code in CSV
        col_text (str): column containing the text
        cat (list): list of labels to use
        """
        self.file = file
        self.sample = pd.read_csv(file)
        if not "codage" in self.sample.columns:
            self.sample["codage"] = None
        self.col_text = col_text
        self.file = file
        self.cat = cat
        self.line = None
        self.line = self.__get_element()
        self.coded_items = []
        self.create_widget()
        return None
    
    def __repr__(self):
        return f"""
Annotation de corpus du fichier {self.file}
Avec les catégories {self.cat}
Actuellement il y a {len(self.sample["codage"].dropna())} éléments codés sur {len(self.sample)}
"""
        

    def __get_element(self):
        """
        Get the element to annotate
        """
        return self.sample[pd.isnull(self.sample["codage"])].index[0]
    
    def __get_text(self):
        """
        Get the content to display
        """
        return self.sample.loc[self.line,self.col_text]
    
    def __add_annotation(self,code):
        """
        Add annotation
        """
        self.sample.loc[self.line,"codage"] = code
        self.coded_items.append(self.line)
        self.line = self.__get_element()

    def __on_button_click(self,b):
        """
        Action if button clicked
        """
        # push the current code and change the current line
        self.__add_annotation(b.description)
        # change the text displayed
        self.texte_widget.value = self.__get_text()
        
    def __back(self,b):
        """
        Reverse operation
        """
        self.line = self.coded_items.pop()
        self.sample.loc[self.line,"codage"] = None
        self.texte_widget.value = self.__get_text()
        return None
    
    def __add_cat(self,a):
        """
        Add a category to cat
        """
        self.cat.append(a.value)
        self.boutons.children += (self.__add_button(a.value),)
        print("Nouveau bouton")
        #self.create_widget()
        return None
        
    def __add_button(self,c):
        """
        Add a button un the HBOX
        """
        b = widgets.Button(description=c)
        b.on_click(self.__on_button_click)
        return b
    
    def create_widget(self):
        """
        Create the widget
        """
        # Text area
        self.texte_widget = widgets.Textarea(value=self.__get_text(),
                                        layout=widgets.Layout(width='700px',height='150px'), 
                                        description='')
        
        # A button for each category
        self.boutons = widgets.HBox([self.__add_button(i) for i in self.cat])
            
        # Back button
        retour = widgets.Button(description="Revenir en arrière",
                                icon="backward")
        retour.on_click(self.__back)
        
        # Field to add a category
        new_cat = widgets.Text(
            value='Nouvelle catégorie',
            placeholder='Type something',
            disabled=False   
            )
        new_cat.on_submit(self.__add_cat)

        
        # Display the widget
        display(retour)
        display(self.texte_widget)
        display(self.boutons)
        display(new_cat)
        
    def save_corpus(self,name="current_sample.csv"):
        """
        Save the corpus
        
        Parameters
        ----------
        
        name (str): name of the CSV file to save the sample with the current state of codes
        """
        self.sample.to_csv(name+".csv")
       
