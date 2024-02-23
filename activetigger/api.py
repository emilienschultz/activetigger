from fastapi import FastAPI, Depends, HTTPException, Header, UploadFile, File, Body, Form, Request
import logging
from typing import Annotated
from datamodels import ProjectModel, ElementModel, SchemesModel, Action, AnnotationModel,SchemeModel, Error
from datamodels import RegexModel, SimpleModelModel, BertModelModel
from server import Server, Project
import functions
import json
from multiprocessing import Process
import time
import pandas as pd
import os

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

# middleware to update elements on events
async def update():
    """
    Function to be executed before each request
    """
    print(f"Updated Value at {time.strftime('%H:%M:%S')}")
    for p in server.projects:
        project = server.projects[p]
        # merge results of subprocesses
        if (project.params.dir / "sbert.parquet").exists():
            log = functions.log_process("sbert", project.params.dir / "log_process.log") 
            log.info("Completing sbert computation")
            df = pd.read_parquet(project.params.dir / "sbert.parquet")
            project.features.add("sbert",df)
            os.remove(project.params.dir / "sbert.parquet")
            logging.info("SBERT embeddings added to project")
        if (project.params.dir / "fasttext.parquet").exists():
            log = functions.log_process("fasttext", project.params.dir / "log_process.log") 
            log.info("Completing fasttext computation")
            df = pd.read_parquet(project.params.dir / "fasttext.parquet")
            project.features.add("fasttext",df)
            os.remove(project.params.dir / "fasttext.parquet")
            print("Adding fasttext embeddings")
            logging.info("FASTTEXT embeddings added to project")


@app.middleware("http")
async def middleware(request: Request, call_next):
    """
    Middleware
    """
    await update()
    response = await call_next(request)
    return response

# ------------
# Dependencies
# ------------

async def get_project(project_name: str) -> ProjectModel:
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
    r = project.get_state()
    print(r)
    return r

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

#async def new_project(project: Annotated[ProjectModel, Depends(get_params)],
@app.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
                      file: Annotated[UploadFile, File()],
                      project_name:str = Form(),
                      col_text:str = Form(),
                      col_id:str = Form(None),
                      n_rows:int = Form(None),
                      embeddings:list = Form(None),
                      n_skip:int = Form(None),
                      langage:str = Form(None),
                      col_tags:str = Form(None),
                      cols_context:list = Form(None)
                      ) -> ProjectModel:
    """
    Load new project
        file (file)
        multiple parameters
    PAS LA SOLUTION LA PLUS JOLIE
    https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522

    """

    # removing None parameters
    params_in = {"project_name":project_name,"col_text":col_text,
              "col_id":col_id,"n_rows":n_rows,"embeddings":embeddings,
              "n_skip":n_skip,"langage":langage,"col_tags":col_tags,
              "cols_context":cols_context}
    params_out = params_in.copy()
    for i in params_in:
        if params_in[i] is None:
            del params_out[i]

    project = ProjectModel(**params_out)

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

@app.post("/projects/delete", dependencies=[Depends(verified_user)])
async def delete_project(project_name:str):
    """
    Delete a project
    """
    r = server.delete_project(project_name)
    return r


# Annotation management
#--------------------

@app.get("/elements/next", dependencies=[Depends(verified_user)])
async def get_next(project: Annotated[Project, Depends(get_project)],
                   scheme:str,
                   selection:str = "deterministic",
                   sample:str = "untagged",
                   user:str = "user",
                   tag:str|None = None) -> ElementModel|Error:
    """
    Get next element
    """
    e = project.get_next(
                        scheme = scheme,
                        selection = selection,
                        sample = sample,
                        user = user,
                        tag = tag
                        )
    print(scheme, selection, sample, user, tag)
    print(e)
    if "error" in e:
        r = Error(**e)
    else:
        r = ElementModel(**e)
    return r

@app.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(project: Annotated[Project, Depends(get_project)],
                            scheme:str,
                            min:int = 0,
                            max:int = 0,
                            mode:str = "all",
                        ):
    
    r = project.schemes.get_table_elements(scheme, min, max, mode)
    return r
    
    

@app.get("/elements/{id}", dependencies=[Depends(verified_user)])
async def get_element(id:str, 
                      project: Annotated[Project, Depends(get_project)]) -> ElementModel:
    """
    Get specific element
    """
    try:
        e = ElementModel(**project.get_element(id))
        return e
    except: # gérer la bonne erreur
        raise HTTPException(status_code=404, detail="Element not found")
    

@app.post("/tags/table", dependencies=[Depends(verified_user)])
async def post_table_tags(project: Annotated[Project, Depends(get_project)],
                          annotation:list[AnnotationModel]):
    """
    Deal with list of tags especially for batch update
    """
    return {"error":"not implemented"}

@app.post("/tags/{action}", dependencies=[Depends(verified_user)])
async def post_tag(action:Action,
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


@app.get("/schemes", dependencies=[Depends(verified_user)])
async def get_schemes(project: Annotated[Project, Depends(get_project)],
                      scheme:str|None = None):
        """
        Available scheme of a project
        """
        if scheme is None:
            return project.schemes.get()
        a = project.schemes.available()
        if scheme in a:
            return {"scheme":a[scheme]}
        return {"error":"scheme not available"}



@app.post("/schemes/{action}", dependencies=[Depends(verified_user)])
async def post_schemes(
                        action:Action,
                        project: Annotated[Project, Depends(get_project)],
                        scheme:SchemeModel
                        ):
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

@app.get("/features", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]):
        """
        Available scheme of a project
        """
        return {"features":list(project.features.map.keys())}

@app.post("/features/add/regex", dependencies=[Depends(verified_user)])
async def post_regex(project: Annotated[Project, Depends(get_project)],
                          regex:RegexModel):
    r = project.add_regex(regex.name,regex.value)
    return r

@app.post("/features/add/{name}", dependencies=[Depends(verified_user)])
async def post_embeddings(project: Annotated[Project, Depends(get_project)],
                          name:str):
    
    # multiple choice:
    # [ ] use multiprocessing with future to deal with cpu-bound tasks
    # [X] launch independant process with actualisation

    log = functions.log_process(name, project.params.dir / "log_process.log") 
    df = project.content[project.params.col_text]
    if name == "sbert":
        log.info("Start computing sbert")
        args = {
                "path":project.params.dir,
                "texts":df,
                "model":"distiluse-base-multilingual-cased-v1"
                }
        process = Process(target=functions.process_sbert, 
                          kwargs = args)
        process.start()
        return {"success":"computing sbert, it could take a few minutes"}
    if name == "fasttext":
        log.info("Start computing fasttext")
        args = {
                "path":project.params.dir,
                "texts":df,
                "model":"/home/emilien/models/cc.fr.300.bin"
                }
        process = Process(target=functions.process_fasttext, 
                          kwargs = args)
        process.start()
        return {"success":"computing fasttext, it could take a few minutes"}


    #    log.info("start sbert computing")
    #    future = server.pool.submit(functions.to_sbert,df)
    #    def callback(x):
    #        r = x.result()
    #        project.features.add("sbert",r)
    #        log.info("computing finished - adding the feature")
    #        return True
    #    future.add_done_callback(callback)
    #    return {"success":"computing sbert, it could take a few minutes"}
    return {"error":"not implemented"}

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
    r = project.simplemodels.available()
    return r


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
    df = project.schemes.get_scheme_data(bert.col_label) #move it elswhere ?
    p = project.bertmodel.start_training_process(name = bert.name,
                                 df=df,
                                 col_text=df.columns[0],
                                 col_label=df.columns[1],
                                 model=bert.model,
                                 params = bert.params,
                                 test_size=bert.test_size)
    server.processes.append(p)
    return {"success":"bert under training"}

    
# add route to test the status of the training