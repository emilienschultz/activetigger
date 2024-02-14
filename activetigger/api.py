from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Body, Form
import logging
from typing import Annotated
from datamodels import ParamsModel, ElementModel, SchemesModel, Action, AnnotationModel,SchemeModel
from datamodels import RegexModel, SimpleModelModel, BertModelModel
from server import Server, Project

logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')



# à terme toute demande de l'app aura les informations de la connexion
# nom de l'utilisateur
# accès autoris
# projet connecté
# Passer l'utilisateur en cookie lors de la connexion ?  https://www.tutorialspoint.com/fastapi/fastapi_cookie_parameters.htm
# unwrapping pydandic object  UserInDB(**user_dict)
# untiliser l'héritage de class & le filtrage


#######
# API #
#######
    
server = Server()
app = FastAPI()

# ------------
# Dependencies
# ------------

async def get_project(project_name: str) -> ParamsModel:
    """
    Fetch existing project associated with the request
    """

    # If project doesn't exist
    if not server.exists(project_name):
        raise HTTPException(status_code=404, detail="Project not found")

    # If the project exist
    if project_name in server.projects:
        # If already loaded
        return server.projects[project_name]
    else:
        # To load
        server.start_project(project_name)            
        return server.projects[project_name]

# TODO : gérer l'authentification de l'utilisateur
async def verified_user(x_token: Annotated[str, Header()]):
    # Cookie ou header ?
    if False:
        raise HTTPException(status_code=400, detail="Invalid user")    

async def get_params(project_name:str = Form(),
                     col_text:str = Form("text"),
                     n_rows:int = Form(2000),
                     col_tags:str = Form(None),
                     embeddings:list = Form([])) -> ParamsModel:
    """
    Collect form data to project params
    """
    p = ParamsModel(project_name=project_name,
                    col_text = col_text,
                    n_rows=n_rows,
                    col_tags=col_tags,
                    embeddings=embeddings)
    return p

# ------
# Routes
# ------


# Projects management
#--------------------

@app.get("/state/{project_name}", dependencies=[Depends(verified_user)])
async def get_state(project: Annotated[Project, Depends(get_project)]):
    """
    Get state of a project
    TODO: a datamodel
    """
    return project.get_state()

@app.get("/projects/{project_name}", dependencies=[Depends(verified_user)])
async def info_project(project_name:str = None):
    """
    Get info on project
    """
    return {project_name:server.db_get_project(project_name)}

@app.get("/projects", dependencies=[Depends(verified_user)])
async def info_all_projects():
    """
    Get all available projects
    """
    return {"existing projects":server.existing_projects()}

@app.post("/project/new", dependencies=[Depends(verified_user)])
async def new_project(project: Annotated[ParamsModel, Depends(get_params)],
                      file: UploadFile = File(),
                      ) -> ParamsModel:
    """
    Load new project
    Parameters:
        file (file)
        n_rows (int)
    """

    # For the moment, only csv
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=422, 
                detail="Only CSV file for the moment")
        
    # Test if project exist
    if server.exists(project.project_name):
        raise HTTPException(status_code=422, 
                detail="Project already exist")

    project = server.create_project(project, file)

    return project
    #return {"success":"project created"}


# Annotation management
#--------------------

@app.get("/element", dependencies=[Depends(verified_user)])
async def get_element(id:str, 
                      project: Annotated[Project, Depends(get_project)],
                      response_model=ElementModel) -> ElementModel:
    """
    Get specific element
    """
    try:
        e = ElementModel(**project.get_element(id))
        return e
    except: # gérer la bonne erreur
        raise HTTPException(status_code=404, detail="Element not found")
    

@app.post("/annotation/{action}", dependencies=[Depends(verified_user)])
async def post_annotation(action:Action,
                          project: Annotated[Project, Depends(get_project)],
                          annotation:AnnotationModel):
    """
    Add, Update, Delete annotations
    """
    if action in ["add","update"]:
        if annotation.tag is None:
            raise HTTPException(status_code=422, 
                detail="Missing a tag")
        return project.schemes.push_tag(annotation.element_id, 
                                        annotation.tag, 
                                        annotation.scheme
                                        )
    if action == "delete":
        project.schemes.delete_tag(annotation.element_id, 
                                   annotation.scheme
                                   )
        return {"success":"label deleted"}
    

# Schemes management
#-------------------


@app.get("/schemes/{project_name}", dependencies=[Depends(verified_user)])
async def get_schemes(project: Annotated[Project, Depends(get_project)]) -> SchemesModel:
        """
        Available scheme of a project
        """
        return project.schemes.get()


@app.get("/annotation/next", dependencies=[Depends(verified_user)])
async def get_next(project: Annotated[Project, Depends(get_project)],
                   scheme:str,
                   mode:str = "deterministic",
                   on:str = "untagged") -> ElementModel:
    """
    Get next element
    """
    e = project.get_next(scheme = scheme,
                         mode = mode,
                         on = on)
        
    return ElementModel(**e)


@app.post("/schemes/{action}", dependencies=[Depends(verified_user)])
async def post_schemes(action:Action,
                          project: Annotated[Project, Depends(get_project)],
                          scheme:SchemeModel):
    """
    Add, Update or Delete scheme
    """
    if action == "add":
        r = project.schemes.add_scheme(scheme)
        return r
    if action == "delete":
        r = project.schemes.delete_scheme(scheme)
        return r
    if action == "update":
        r = project.schemes.update_scheme(scheme)
        return r
    
    return {"error":"wrong route"}


# Features management
#--------------------

@app.get("/features/{project_name}", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]):
        """
        Available scheme of a project
        """
        return {"features":list(project.features.map.keys())}

@app.post("/features/regex", dependencies=[Depends(verified_user)])
async def post_regex(project: Annotated[Project, Depends(get_project)],
                          regex:RegexModel):
    r = project.add_regex(regex.name,regex.value)
    return r


@app.post("/features/embeddings/{name}", dependencies=[Depends(verified_user)])
async def post_embeddings(project: Annotated[Project, Depends(get_project)],
                          name:str):
    r = project.compute_embeddings(name)
    return r

@app.post("/features/delete", dependencies=[Depends(verified_user)])
async def delete_feature(project: Annotated[Project, Depends(get_project)],
                     name:str):
    r = project.features.delete(name)
    return r


# Models management
#------------------

@app.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(project: Annotated[Project, Depends(get_project)]):
    """
    Simplemodel parameters
    """
    return project.simplemodel.get_params()


@app.post("/models/simplemodel", dependencies=[Depends(verified_user)])
async def post_simplemodel(project: Annotated[Project, Depends(get_project)],
                           simplemodel:SimpleModelModel):
    """
    Compute simplemodel
    """
    r = project.update_simplemodel(simplemodel)
    return r

@app.get("/models/bert", dependencies=[Depends(verified_user)])
async def get_bert(project: Annotated[Project, Depends(get_project)]):
    """
    bert parameters
    """
    return {"error":"Pas implémenté"}#project.bertmodel.get_params()


@app.post("/models/bert", dependencies=[Depends(verified_user)])
async def post_bert(project: Annotated[Project, Depends(get_project)],
                     bert:BertModelModel):
    """ 
    Compute bertmodel
    """
    # add content (put it inside bert ?)
    df = project.schemes.get_annotations(bert.col_label)

    project.bertmodel.train_bert(name = bert.name,
                                 df = df,
                                 col_label = df.columns[0],
                                 col_text = df.columns[1],
                                 model_name = bert.model_name,
                                 params = bert.params,
                                 test_size = bert.test_size)
    
    return {"success":"bert trained"}

