from fastapi import FastAPI, Depends, HTTPException,status, Header, UploadFile, File, Query, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import logging
from typing import Annotated, List
from multiprocessing import Process
import time
import pandas as pd
import os
from jose import JWTError
import importlib

from activetigger.datamodels import ProjectModel, ElementModel, TableElementsModel, Action, AnnotationModel, SchemeModel, Error, ProjectionModel, User, Token, RegexModel, SimpleModelModel, BertModelModel, ParamsModel
from activetigger.server import Server, Project
import activetigger.functions as functions

logging.basicConfig(filename='log.log', 
                    encoding='utf-8', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


# General comments
# - all post are logged
# - header identification with token
# - username is in the header

#######
# API #
#######

# start the backend server
server = Server()

# start the app
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")
    server.executor.shutdown(cancel_futures=True, wait = False) #clean async multiprocess

app = FastAPI(lifespan=lifespan)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# middleware to update elements on events
async def update():
    """
    Function to be executed before each request
    """
    print(f"Updated Value at {time.strftime('%H:%M:%S')}")
    for p in server.projects:
        project = server.projects[p]
        # computing embeddings
        if (project.params.dir / "sbert.parquet").exists():
            df = pd.read_parquet(project.params.dir / "sbert.parquet") # load data TODO : bug potentiel lié à la temporalité
            project.features.add("sbert",df) # add to the feature manager
            if "sbert" in project.features.training:
                project.features.training.remove("sbert") # remove from pending processes
            os.remove(project.params.dir / "sbert.parquet") # clean the files
            logging.info("SBERT embeddings added to project") # log
        if (project.params.dir / "fasttext.parquet").exists():
            df = pd.read_parquet(project.params.dir / "fasttext.parquet")
            project.features.add("fasttext",df) 
            if "fasttext" in project.features.training:
                project.features.training.remove("fasttext") 
            os.remove(project.params.dir / "fasttext.parquet")
            print("Adding fasttext embeddings")
            logging.info("FASTTEXT embeddings added to project")
        if (project.params.dir / "dfm.parquet").exists():
            df = pd.read_parquet(project.params.dir / "dfm.parquet")
            project.features.add("dfm",df) 
            if "dfm" in project.features.training:
                project.features.training.remove("dfm") 
            os.remove(project.params.dir / "dfm.parquet")
            print("Adding dfm embeddings")
            logging.info("Dfm embeddings added to project")
        
        # joining projection process
        for u in project.features.available_projections:
            if ("future" in project.features.available_projections[u]):
                if project.features.available_projections[u]["future"].done():
                    df = project.features.available_projections[u]["future"].result()
                    project.features.available_projections[u]["data"] = df
                    del project.features.available_projections[u]["future"]
                    print("Adding projection data")

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

async def verified_user(Authorization: Annotated[str, Header()],
                        username: Annotated[str, Header()]):
    """
    Test if the user is a user authentified
    For the moment, only that there is a well formed user
    TODO : a real test here
    """
    #print(f"{username} does an action")
    if False:
        raise HTTPException(status_code=400, detail="Invalid user")    

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Get current user from the token in headers
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = server.decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = server.get_user(name=username)
    if user is None:
        raise credentials_exception
    return user

# ------
# Routes
# ------

@app.get("/", response_class=HTMLResponse)
async def welcome():
    """
    Welcome page
    """
    data_path = importlib.resources.files("activetigger")
    with open(data_path / "html/welcome.html","r") as f:
        return f.read()

# Users
#------

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token:
    """
    Route to create token for user from form data
    """
    user = server.authenticate_user(form_data.username, form_data.password)
    if "error" in user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = server.create_access_token(
        data={"sub": user.username}, 
        expires_min=60)
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]):
    """
    Information on current user
    """
    return current_user

@app.post("/users/create", dependencies=[Depends(verified_user)])
async def create_user(username:str = Query(),
                      password:str = Query(),
                      projects:str = Query(None)):
    """
    Create user
    """
    r = server.add_user(username, password, projects)
    return r

@app.post("/users/delete", dependencies=[Depends(verified_user)])
async def delete_user(username:str = Query()):
    """
    Delete user
    """
    r = server.delete_user(username)
    return r

# Projects management
#--------------------

@app.get("/state/{project_name}", dependencies=[Depends(verified_user)])
async def get_state(project: Annotated[Project, Depends(get_project)]):
    """
    Get state of a project
    """
    r = project.get_state()
    return r

@app.get("/description", dependencies=[Depends(verified_user)])
async def get_description(project: Annotated[Project, Depends(get_project)],
                          scheme: str|None = None,
                          user:str|None = None):
    """
    Get state for a specific project/scheme/user
    """
    r = project.get_description(scheme = scheme, user = user)
    return r

@app.get("/projects/{project_name}", dependencies=[Depends(verified_user)])
async def info_project(project_name:str|None = None):
    """
    Get info on project
    """
    return {project_name:server.db_get_project(project_name)}


@app.get("/server")
async def info_server():
    """
    Get info on the server (no validation needed)
    """
    r = {
        "projects":server.existing_projects(),
        "users":server.existing_users()
        }
    return r

@app.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
                      username: Annotated[str, Header()],
                      file: Annotated[UploadFile, File()],
                      project_name:str = Form(),
#                      user:str = Form(),
                      col_text:str = Form(),
                      col_id:str = Form(),
                      col_label:str = Form(None),
                      cols_context:List[str] = Form(None),
                      n_train:int = Form(),
                      n_test:int = Form(),
                      embeddings:list = Form(None),
                      n_skip:int = Form(None),
                      langage:str = Form(None),
                      ) -> ProjectModel|Error:
    """
    Load new project
        file (file)
        multiple parameters
    PAS LA SOLUTION LA PLUS JOLIE
    https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522

    """

    # removing None parameters
    params_in = {
        "project_name":project_name,
        "user":username,         
        "col_text":col_text,
        "col_id":col_id,
        "n_train":n_train,
        "n_test":n_test,
        "embeddings":embeddings,
        "n_skip":n_skip,
        "langage":langage,
        "col_label":col_label,
        "cols_context":cols_context
        }
    
    params_out = params_in.copy()
    for i in params_in:
        if params_in[i] is None:
            del params_out[i]

    project = ProjectModel(**params_out)

    # For the moment, only csv
    if not file.filename.endswith('.csv'):
        return Error(error = "Only CSV file for the moment")
        
    # Test if project exist
    if server.exists(project.project_name):
        return Error(error = "Project already exist")

    project = server.create_project(project, file)

    # log action
    server.log_action(username, "create project", params_in["project_name"])

    return project

@app.post("/projects/delete", dependencies=[Depends(verified_user)])
async def delete_project(username: Annotated[str, Header()],
                         project_name:str):
    """
    Delete a project
    TODO : user authorization
    """
    r = server.delete_project(project_name)
    server.log_action(username, "delete project", project_name)
    return r

# Annotation management
#--------------------

@app.get("/elements/next", dependencies=[Depends(verified_user)])
async def get_next(project: Annotated[Project, Depends(get_project)],
                   username: Annotated[str, Header()],
                   scheme:str,
                   selection:str = "deterministic",
                   sample:str = "untagged",
                   #user:str = "user",
                   tag:str|None = None,
                   frame:list[float]|None = Query(None)) -> ElementModel|Error:
    """
    Get next element
    """
    e = project.get_next(
                        scheme = scheme,
                        selection = selection,
                        sample = sample,
                        user = username,
                        tag = tag,
                        frame = frame
                        )
    if "error" in e:
        r = Error(**e)
    else:
        r = ElementModel(**e)
    return r


@app.get("/elements/projection/current", dependencies=[Depends(verified_user)])
async def get_projection(project: Annotated[Project, Depends(get_project)],
                         username: Annotated[str, Header()],
                         #user:str, 
                         scheme:str|None):
    """
    Get projection data if computed
    """
    if username in project.features.available_projections:
        if not "data" in project.features.available_projections[username]:
            return {"status":"Still computing"}
        if scheme is None:
            return {"data":project.features.available_projections[username]["data"].fillna("NA").to_dict()}
        else: # add the labels of the scheme
            data = project.features.available_projections[username]["data"]
            df = project.schemes.get_scheme_data(scheme)
            data["labels"] = df["labels"]
            return {"data":data.fillna("NA").to_dict()}

    return {"error":"There is no projection available"}

@app.post("/elements/projection/compute", dependencies=[Depends(verified_user)])
async def compute_projection(project: Annotated[Project, Depends(get_project)],
                         username: Annotated[str, Header()],
                         #user:str,
                         projection:ProjectionModel):
    """
    Start projection computation
    Dedicated process, end with a file on the project
    projection__user.parquet
    TODO : très moche comme manière de faire, à reprendre
    """
    if len(projection.features) == 0:
        return {"error":"No feature"}
    
    name = f"projection__{username}"
    features = project.features.get(projection.features)
    args = {
            "features":features,
            "params":projection.params
            }

    if projection.method == "umap":
        future_result = server.executor.submit(functions.compute_umap, **args)
        project.features.available_projections[username] = {
                                                        "params":projection,
                                                        "method":"umap",
                                                        "future":future_result
                                                        }
        return {"success":"Projection umap under computation"}
    if projection.method == "tsne":
        future_result = server.executor.submit(functions.compute_tsne, **args)
        project.features.available_projections[username] = {
                                                        "params":projection,
                                                        "method":"tsne",
                                                        "future":future_result
                                                        }
        return {"success":"Projection tsne under computation"}
    return {"error":"This projection is not available"}

@app.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(project: Annotated[Project, Depends(get_project)],
                            scheme:str,
                            min:int = 0,
                            max:int = 0,
                            mode:str = "all",
                        ):
    """
    Get table of elements
    """
    r = project.schemes.get_table(scheme, min, max, mode)
    return r.fillna("NA")
    
@app.post("/elements/table", dependencies=[Depends(verified_user)])
async def post_list_elements(project: Annotated[Project, Depends(get_project)],
                            username: Annotated[str, Header()],
                            #user:str,
                            table:TableElementsModel
                            ):
    r = project.schemes.push_table(table = table, 
                                   user = username)
    server.log_action(username, "update data table", project.name)
    return r

@app.get("/elements/{element_id}", dependencies=[Depends(verified_user)])
async def get_element(project: Annotated[Project, Depends(get_project)],
                      username: Annotated[str, Header()],
                      element_id:str,
                      scheme:str) -> ElementModel:
    """
    Get specific element
    """
    print(element_id)
    try:
        e = ElementModel(**project.get_element(element_id, scheme=scheme, user=username))
        return e
    except: # gérer la bonne erreur
        raise HTTPException(status_code=404, detail=f"Element {element_id} not found")
    

@app.post("/tags/{action}", dependencies=[Depends(verified_user)])
async def post_tag(action:Action,
                   username: Annotated[str, Header()],
                   project: Annotated[Project, Depends(get_project)],
                   annotation:AnnotationModel):
    """
    Add, Update, Delete annotations
    Comment : 
    - For the moment add == update
    """
    if action in ["add","update"]:
        if annotation.tag is None:
            raise HTTPException(status_code=422, 
                detail="Missing a tag")
        r = project.schemes.push_tag(annotation.element_id, 
                                    annotation.tag, 
                                    annotation.scheme,
                                    username
                                    #annotation.user
                                    )
        server.log_action(username, f"push annotation {annotation.element_id}", project.name)
        return r

    if action == "delete":
        project.schemes.delete_tag(annotation.element_id, 
                                   annotation.scheme,
                                   username
                                   #annotation.user
                                   ) # add user deletion
        server.log_action(username, f"delete annotation {annotation.element_id}", project.name)
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


@app.post("/schemes/label/add", dependencies=[Depends(verified_user)])
async def add_label(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    scheme:str,
                    label:str,
                    #user:str
                    ):
    """
    Add a label to a scheme
    """
    r = project.schemes.add_label(label, scheme, username)
    server.log_action(username, f"add label {label} to {scheme}", project.name)
    return r

@app.post("/schemes/label/delete", dependencies=[Depends(verified_user)])
async def delete_label(project: Annotated[Project, Depends(get_project)],
                       username: Annotated[str, Header()],
                    scheme:str,
                    label:str,
                    #user:str
                    ):
    """
    Remove a label from a scheme
    """
    r = project.schemes.delete_label(label, scheme, username)
    server.log_action(username, f"delete label {label} to {scheme}", project.name)
    return r


@app.post("/schemes/{action}", dependencies=[Depends(verified_user)])
async def post_schemes(username: Annotated[str, Header()],
                        project: Annotated[Project, Depends(get_project)],
                        action:Action,
                        scheme:SchemeModel
                        ):
    """
    Add, Update or Delete scheme
    TODO : user dans schememodel, necessary ?
    """
    if action == "add":
        print("add")
        r = project.schemes.add_scheme(scheme)
        server.log_action(username, f"add scheme {scheme.name}", project.name)
        return r
    if action == "delete":
        r = project.schemes.delete_scheme(scheme)
        server.log_action(username, f"delete scheme {scheme.name}", project.name)
        return r
    if action == "update":
        r = project.schemes.update_scheme(scheme.name, scheme.tags, username)
        server.log_action(username, f"update scheme {scheme.name}", project.name)
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
                     username: Annotated[str, Header()],
                          regex:RegexModel):
    """
    Add a regex
    """
    r = project.add_regex(regex.name,regex.value)
    server.log_action(username, f"add regex {regex.name}", project.name)
    return r

@app.post("/features/add/{name}", dependencies=[Depends(verified_user)])
async def post_embeddings(project: Annotated[Project, Depends(get_project)],
                          username: Annotated[str, Header()],
                          name:str,
                          params:ParamsModel
                          ):
    """
    Compute features :
    common logic : call a function in a specific process, 
    create a file, and then updated
    TODO : refactorize the function + merge with regex
    """
    if name in project.features.training:
        return {"error":"This feature is already in training"}
    
    df = project.content[project.params.col_text]
    if name == "sbert":
        args = {
                "path":project.params.dir,
                "texts":df,
                "model":"distiluse-base-multilingual-cased-v1"
                }
        process = Process(target=functions.process_sbert, 
                          kwargs = args)
        process.start()
        project.features.training.append(name)
        server.log_action(username, f"Compute feature sbert", project.name)
        return {"success":"computing sbert, it could take a few minutes"}
    if name == "fasttext":
        args = {
                "path":project.params.dir,
                "texts":df,
                "model":"/home/emilien/models/cc.fr.300.bin"
                }
        process = Process(target=functions.process_fasttext, 
                          kwargs = args)
        process.start()
        project.features.training.append(name)
        server.log_action(username, f"Compute feature fasttext", project.name)
        return {"success":"computing fasttext, it could take a few minutes"}
    if name == "dfm":
        args = params.params
        args["texts"] = df
        args["path"] = project.params.dir
        process = Process(target=functions.process_dfm, 
                          kwargs = args)
        process.start()
        project.features.training.append(name)
        server.log_action(username, f"Compute feature dfm", project.name)
        return {"success":"computing dfm, it could take a few seconds"}

    return {"error":"not implemented"}

@app.post("/features/delete/{name}", dependencies=[Depends(verified_user)])
async def delete_feature(project: Annotated[Project, Depends(get_project)],
                        username: Annotated[str, Header()],
                        #user:str,
                        name:str):
    """
    Delete a specific feature
    """
    r = project.features.delete(name)
    server.log_action(username, f"delete feature {name}", project.name)
    return r


# Models management
#------------------

@app.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(project: Annotated[Project, Depends(get_project)]):
    """
    Simplemodel parameters
    """
    r = project.simplemodels.available()
    print(type(r))
    return r


@app.post("/models/simplemodel", dependencies=[Depends(verified_user)])
async def post_simplemodel(project: Annotated[Project, Depends(get_project)],
                           username: Annotated[str, Header()],
                           simplemodel:SimpleModelModel):
    """
    Compute simplemodel
    TODO : user out of simplemodel
    """
    r = project.update_simplemodel(simplemodel)
    return r

@app.get("/models/bert", dependencies=[Depends(verified_user)])
async def get_bert(project: Annotated[Project, Depends(get_project)],
                   name:str):
    """
    Bert parameters and statistics
    """
    b = project.bertmodels.get(name, lazy= True)
    if b is None:
        return {"error":"Bert model does not exist"}
    r =  b.informations()
    print(r)
    return r

@app.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict(project: Annotated[Project, Depends(get_project)],
                  username: Annotated[str, Header()],
                     model_name:str,
                     #user:str,
                     data:str = "all"):
    """
    Start prediction with a model
    TODO : scope data
    """
    print("start predicting")
    df = project.content[["text"]]
    r = project.bertmodels.start_predicting_process(name = model_name,
                                                    df = df,
                                                    col_text = "text",
                                                    user = username)
    server.log_action(username, f"predict bert {model_name}", project.name)
    return r

@app.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                     bert:BertModelModel):
    """ 
    Compute bertmodel
    TODO : gestion du nom du projet/scheme à la base du modèle
    """
    print("start bert training")
    df = project.schemes.get_scheme_data(bert.scheme, complete = True) #move it elswhere ?
    df = df[[project.params.col_text, "labels"]].dropna() #remove non tag data
    r = project.bertmodels.start_training_process(
                                name = bert.name,
                                user = username,
                                scheme = bert.scheme,
                                df = df,
                                col_text=df.columns[0],
                                col_label=df.columns[1],
                                base_model=bert.base_model,
                                params = bert.params,
                                test_size=bert.test_size
                                )
    server.log_action(username, f"train bert {bert.name}", project.name)
    return r

@app.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    #user:str
                     ):
    r = project.bertmodels.stop_user_training(username)
    server.log_action(username, f"stop bert training", project.name)
    return r

@app.post("/models/bert/delete", dependencies=[Depends(verified_user)])
async def delete_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    bert_name:str):
    """
    Delete trained bert model
    """
    r = project.bertmodels.delete(bert_name)
    server.log_action(username, f"delete bert model {bert_name}", project.name)
    return r

@app.post("/models/bert/rename", dependencies=[Depends(verified_user)])
async def save_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                     former_name:str,
                     new_name:str,
                     #user:str
                     ):
    r = project.bertmodels.rename(former_name, new_name)
    server.log_action(username, f"rename bert model {former_name} - {new_name}", project.name)
    return r


# Export elements
#----------------

@app.get("/export/data", dependencies=[Depends(verified_user)])
async def export_data(project: Annotated[Project, Depends(get_project)],
                      scheme:str,
                      format:str):
    name, path = project.export_data(format=format, scheme=scheme)
    return FileResponse(path, filename=name)

@app.get("/export/features", dependencies=[Depends(verified_user)])
async def export_features(project: Annotated[Project, Depends(get_project)],
                          features:list = Query(),
                          format:str = Query()):
    name, path = project.export_features(features = features, format=format)
    return FileResponse(path, filename=name)

@app.get("/export/prediction", dependencies=[Depends(verified_user)])
async def export_prediction(project: Annotated[Project, Depends(get_project)],
                          format:str = Query(),
                          name:str = Query()):
    name, path = project.bertmodels.export_prediction(name = name, format=format)
    return FileResponse(path, filename=name)

@app.get("/export/bert", dependencies=[Depends(verified_user)])
async def export_bert(project: Annotated[Project, Depends(get_project)],
                          name:str = Query()):
    name, path = project.bertmodels.export_bert(name = name)
    return FileResponse(path, filename=name)
