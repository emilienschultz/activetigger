from project import Project

import logging
logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

class Server():
    """
    Backend
    """

    def __init__(self):
        """
        Start the server
        """
        self.projects = {}
        logging.warning('Still under development')
        logging.info("Start server")

    def start_project(self,project_name, **kwargs):
        """
        Initialize a project with a project_name
        """
        self.projects[project_name] = Project(project_name, **kwargs)

    # ENDPOINTS (in future FASTAPI)

    def get(self, req):
        """
        Get data from server
        """
        logging.info(f"Get request from frontend {req}")

        # check credentials and parameters

        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        
        p = self.projects[req["project_name"]]

        # deal responses

        if req["type"] == "next" :
            req = p.get_next(req["mode"])
            return req

        if req["type"] == "element" :
            req = p.get_element(req["element_id"])
            return req
        
        if req["type"] == "schemes":
            return {
                    "type":"schemes",
                    "content":p.schemes.dump()
                    }
        
        if req["type"] == "simplemodel":
            return {

                "type":"simplemodel",
                "content":p.simplemodel.get_params()
            }

        #if req["type"] == "params":
        #    return p.params
        
        return {}
    
    def post(self, req):
        """
        Deal post data
        """
        logging.info(f"Post request from frontend {req}")

        # check credentials and parameters

        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        
        p = self.projects[req["project_name"]]

        # manage post request
        if req["type"] == "label" :
            p.add_label(req["content"]["element_id"]["element_id"],req["content"]["label"])
            return {"add":"success"}
        
        if req["type"] == "delete_label":
            p.delete_label(req["content"]["element_id"])
            return {"delete":"success"}            

        if req["type"] == "update_schemes":
            p.update_schemes(req["content"])
            return {"update_schemes":"success"}
        
        if req["type"] == "update_simplemodel":
            # train a new simple model
            # TODO : il y a un problème quand le modele est nul
            # car il n'y a pas encore de sélection de label/embeddings
            p.simplemodel.update(req["content"])
            return {"update_simplemodel":"success"}