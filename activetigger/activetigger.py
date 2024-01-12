# PyActiveTigger

import server
import widget

__version__ = "0.0.1"

class ActiveTigger():
    def __init__(self,
                 project_name:str,
                 file:str=None,
                 n_rows:int=0,
                 cat:list = None,
                 col_text:str = None,
                 **kwargs
                 ):
        # start server
        self.s = server.Server()

        # launch project
        self.s.start_project(project_name,
                        file=file,
                        n_rows=n_rows,
                        cat=cat,
                        col_text=col_text,
                        **kwargs)
        
        # create frontend
        self.w = widget.Widget()

        # connect
        self.w.connect(self.s,project_name)

    def __repr__(self):
        """
        Summary of the current widget
        """
        return None
        