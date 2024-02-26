# PyActiveTigger

import server
import widget

__version__ = "0.0.1"

if __name__ == "__main__":
    #launch stuff

class ActiveTigger():
    def __init__(self,
                 project_name:str,
                 file:str|None=None,
                 n_rows:int=0,
                 cat:list|None = None,
                 col_text:str|None = None,
                 **kwargs
                 ):
        # start server
        self.s: Server = Server()

        # launch project
        self.s.start_project(project_name,
                        file=file,
                        n_rows=n_rows,
                        cat=cat,
                        col_text=col_text,
                        **kwargs)
        
        # create frontend
        self.w: Widget = Widget()

        # connect
        self.w.connect(self.s,project_name)
        