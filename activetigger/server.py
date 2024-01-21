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
    def __init__(self) -> None:
        """
        Start the server
        """
        self.projects: dict = {}
        logging.warning('Still under development')
        logging.info("Start server")

    def start_project(self,project_name, **kwargs):
        """
        Initialize a project with a project_name
        """
        self.projects[project_name] = Project(project_name, **kwargs)

    # ENDPOINTS (in future FASTAPI)

    def check_credentials(self, req: dict) -> Project|dict:
        """
        Check credentials and access to project
        #TODO: users
        """
        if not req["project_name"] in self.projects:
            return {"error":"project doesn't exist"}
        p = self.projects[req["project_name"]]
        return p

    def get(self, req) -> dict:
        """
        Get data from server
        """
        logging.info(f"Get request : {req}")
        p:Project|dict = self.check_credentials(req)
        if type(p) is dict:
            return p

        if req["type"] == "next" :
            req = p.get_next(mode = req["content"]["mode"]["mode"], 
                             on = req["content"]["mode"]["on"])
            return req
        
        if req["type"] == "state" :
            req = p.get_state()
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
                
        return {"error":"request not found"}
    
    def post(self, req:dict) -> dict:
        """
        Manage post requests
        """
        logging.info(f"Post request : {req}")
        p:Project|dict = self.check_credentials(req)
        if type(p) is dict:
            return p

        if req["type"] == "label" :
            p.add_label(req["content"]["element_id"],
                        req["content"]["label"])
            return {"add_label":"success"}
        
        if req["type"] == "delete_label":
            p.delete_label(req["content"]["element_id"])
            return {"delete_label":"success"}            

        if req["type"] == "update_schemes":
            p.update_schemes(req["content"])
            return {"update_schemes":"success"}
        
        if req["type"] == "simplemodel":
            # train a new simple model
            return p.update_simplemodel(req["content"])
        
        return {"error":"request not found"}
