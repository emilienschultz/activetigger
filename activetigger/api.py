import time
from fastapi import FastAPI, Depends, HTTPException,status, Header, UploadFile, File, Query, Form, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from contextlib import asynccontextmanager
import logging
from typing import Annotated, List
from jose import JWTError
import importlib
from pydantic import ValidationError
from activetigger.server import Server, Project
import activetigger.functions as functions
from activetigger.datamodels import ProjectModel, TableElementsModel, Action, AnnotationModel,\
      SchemeModel, ResponseModel, ProjectionModel, User, Token, RegexModel, SimpleModelModel, BertModelModel, ParamsModel,\
      UmapParams, TsneParams, NextModel, ZeroShotModel

logging.basicConfig(filename='log_server.log', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger('api')


# General comments
# - all post are logged
# - header identification with token
# - username is in the header

#######
# API #
#######

# start the backend server
logger.info("Start API")
server = Server()
timer = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Frame the execution of the api
    """
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")
    server.queue.close()

app = FastAPI(lifespan=lifespan)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def check_queue(timer) -> None:
    """
    Call different functions to update the server
    Max one time per second
    """
    # max one update per second to avoid excessive action
    step = 1
    if (time.time()-timer)<step:
        return None
    timer = time.time()
    
    # check the queue
    server.queue.check()

    # update processes for active projects
    # and make actions 
    to_del = []
    for p in server.projects:
        project = server.projects[p]

        # if project existing since one day
        if (timer - project.starting_time) > 86400:
            to_del.append(p)

        project.features.update_processes()
        project.simplemodels.update_processes()
        predictions = project.bertmodels.update_processes()

        # action : if predictions, add them as features
        if len(predictions)>0:
            for f in predictions:
                project.features.add(f, predictions[f])

    # delete old project (they will be loaded if needed)
    for p in to_del:
        del server.projects[p]
    
@app.middleware("http")
async def middleware(request: Request, call_next):
    """
    Middleware to take care of completed processes
    Executed at each action on the server
    """
    await check_queue(timer)
    response = await call_next(request)
    return response

# ------------
# Dependencies
# ------------

async def get_project(project_name: str) -> ProjectModel:
    """
    Fetch existing project
    - if already loaded, return it
    - if not loaded, load it first
    """

    # if project doesn't exist
    if not server.exists(project_name):
        return ResponseModel(status="error", message="Project not found")

    # if the project exist
    if project_name in server.projects:
        return server.projects[project_name]
    else:
        server.start_project(project_name)            
        return server.projects[project_name]

# define credential exception
credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

async def verified_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Test if the user is a user authentified
    """
    try:
        payload = server.decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            return ResponseModel(status="error", message="Could not validate credential")
    except JWTError:
        return ResponseModel(status="error", message="Could not validate credential")
    user = server.get_user(name=username)
    if user is None:
        return ResponseModel(status="error", message="Could not validate credential")

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    """
    Get current user from the headers
    """
    try:
        payload = server.decode_access_token(token)
        username: str = payload.get("sub")
        if username is None:
            return ResponseModel(status="error", message="Could not validate credential")
    except JWTError:
        return ResponseModel(status="error", message="Could not validate credential")
    user = server.get_user(name=username)
    if user is None:
        return ResponseModel(status="error", message="Could not validate credential")
    return user

# ------
# Routes
# ------

@app.get("/", response_class=HTMLResponse)
async def welcome() -> str:
    """
    Welcome page at the root path for the API
    """
    data_path = importlib.resources.files("activetigger")
    with open(data_path / "html/welcome.html","r") as f:
        r = f.read()
    return r

@app.get("/documentation")
async def get_documentation()  -> ResponseModel:
    """
    Path for documentation 
    """
    data = {
            "Credits":["Julien Boelaert", "Étienne Ollion", "Émilien Schultz"],
            "Contact":"emilien.schultz@ensae.fr",
            "Page":"https://github.com/emilienschultz/pyactivetigger",
            "Documentation":"To write ...."
            }
    r = ResponseModel(status="success", data=data)
    return r

# Users
#------

@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]) -> Token|ResponseModel:
    """
    Route to authentificate user and return token
    """
    user = server.authenticate_user(form_data.username, form_data.password)
    if "error" in user:
        r = ResponseModel(status="error", message=user["error"])
        return r
    access_token = server.create_access_token(
            data={"sub": user.username}, 
            expires_min=60)
    r = Token(access_token=access_token, token_type="bearer")
    return r

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_user)]) -> ResponseModel:
    """
    Information on current user
    """
    r = ResponseModel(status="success", data=current_user)
    return r

@app.get("/users", dependencies=[Depends(verified_user)])
async def existing_users() -> ResponseModel:
    """
    Get users information
    """
    data = {
        "users":server.existing_users()
        }
    r = ResponseModel(status="success", data=data)
    return r

@app.post("/users/create", dependencies=[Depends(verified_user)])
async def create_user(username:str = Query(),
                      password:str = Query(),
                      projects:str = Query(None)) -> ResponseModel:
    """
    Create user
    """
    r = server.add_user(username, password, projects)
    if "success" in r:
        return ResponseModel(status="success", message=r["success"])
    else:
        return ResponseModel(status="error", message=r["success"])

@app.post("/users/delete", dependencies=[Depends(verified_user)])
async def delete_user(username:str = Query()) -> ResponseModel:
    """
    Delete user
    """
    r = server.delete_user(username)
    if "success" in r:
        return ResponseModel(status="success", message=r["success"])
    else:
        return ResponseModel(status="error", message=r["success"])

# Projects management
#--------------------

@app.get("/state/{project_name}", dependencies=[Depends(verified_user)])
async def get_state(project: Annotated[Project, Depends(get_project)]) -> ResponseModel:
    """
    Get the state of a specific project
    """
    data = project.get_state()
    r = ResponseModel(status="success", data=data)
    return r

@app.get("/queue")
async def get_queue() -> ResponseModel:
    """
    Get the state of the server queue
    """
    r = server.queue.state()
    return ResponseModel(status="success", data=r)

@app.get("/description", dependencies=[Depends(verified_user)])
async def get_description(project: Annotated[Project, Depends(get_project)],
                          scheme: str|None = None,
                          user: str|None = None)  -> ResponseModel:
    """
    Get the state for a specific project/scheme/user
    """
    data = project.get_description(scheme = scheme, 
                                user = user)
    if "error" in data:
        r = ResponseModel(status="error", message=data["error"])
    else:
        r = ResponseModel(status="success", data=data)
    return r

@app.get("/server")
async def info_server() -> ResponseModel:
    """
    Get general informations on the server 
    (no validation needed)
    """
    data = {
        "projects":server.existing_projects(),
        "users":server.existing_users()
        }
    r = ResponseModel(status="success", data=data)
    return r

@app.post("/projects/testdata", dependencies=[Depends(verified_user)])
async def add_testdata(project: Annotated[Project, Depends(get_project)],
                      username: Annotated[str, Header()],
                      file: Annotated[UploadFile, File()],
                      col_text:str = Form(),
                      col_id:str = Form(),
                      n_test:int = Form())-> ResponseModel:
    """
    Add test dataset
    TODO : operation at the server level
    """

    r = project.add_testdata(file, col_text, col_id, n_test)
    # log action
    server.log_action(username, "add testdata project", project.name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])   

    # if success, update also parameters of the project
    server.set_project_parameters(project.params)

    return ResponseModel(status="success", message=r["success"])  

@app.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
                      username: Annotated[str, Header()],
                      file: Annotated[UploadFile, File()],
                      project_name:str = Form(),
                      col_text:str = Form(),
                      col_id:str = Form(),
                      col_label:str = Form(None),
                      cols_context:List[str] = Form(None),
                      n_train:int = Form(),
                      n_test:int = Form(),
                      cols_test:List[str] = Form(None),
                      embeddings:list = Form(None),
                      n_skip:int = Form(None),
                      language:str = Form(None),
                      ) -> ResponseModel:
    """
    Load new project
        file (file)
        multiple parameters
    Actuellement, pas la solution la plus jolie
    mais je n'arrive pas à mettre tous les éléments dans un objet (query + form + file)
    https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request/70640522#70640522
    """

    # grouping informations
    params_in = {
        "project_name":project_name,
        "user":username,         
        "col_text":col_text,
        "col_id":col_id,
        "n_train":n_train,
        "n_test":n_test,
        "cols_test":cols_test,
        "embeddings":embeddings,
        "n_skip":n_skip,
        "language":language,
        "col_label":col_label,
        "cols_context":cols_context
        }
    
    # removing None parameters
    params = {i:params_in[i] for i in params_in if params_in[i] is not None}
    project = ProjectModel(**params)

    # format of the files (only CSV for the moment)
    if not file.filename.endswith('.csv'):
        return ResponseModel(status = "error", message = "Only CSV file for the moment")
        
    # test if project name already exists
    if server.exists(project.project_name):
        return ResponseModel(status = "error", message = "Project already exist")

    # create the project
    server.create_project(project, file)

    # log action
    server.log_action(username, "create project", params_in["project_name"])

    return ResponseModel(status = "success")

@app.post("/projects/delete", dependencies=[Depends(verified_user)])
async def delete_project(username: Annotated[str, Header()],
                         project_name:str) -> ResponseModel:
    """
    Delete a project
    """
    r = server.delete_project(project_name)
    server.log_action(username, "delete project", project_name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    else:
        return ResponseModel(status="success",  message=r["success"])

@app.post("/elements/next", dependencies=[Depends(verified_user)])
async def get_next(project: Annotated[Project, Depends(get_project)],
                   username: Annotated[str, Header()],
                   next:NextModel ) -> ResponseModel:
    """
    Get next element
    """
    r = project.get_next(
                        scheme = next.scheme,
                        selection = next.selection,
                        sample = next.sample,
                        user = username,
                        tag = next.tag,
                        history=next.history,
                        frame = next.frame
                        )
    print(r)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    return ResponseModel(status="success", data = r)

@app.get("/elements/projection/current", dependencies=[Depends(verified_user)])
async def get_projection(project: Annotated[Project, Depends(get_project)],
                         username: Annotated[str, Header()],
                         scheme:str|None) -> ResponseModel:
    """
    Get projection data if computed
    """
    if not username in project.features.projections:
        return ResponseModel(status="error", message="There is no projection available")

    if not "data" in project.features.projections[username]:
        return ResponseModel(status="waiting", message="Still computing")
    
    if scheme is None:
        data = project.features.projections[username]["data"].fillna("NA").to_dict()
    else:
        # TODO : add texts
        data = project.features.projections[username]["data"]
        df = project.schemes.get_scheme_data(scheme, complete = True)
        data["labels"] = df["labels"]
        data["texts"] = df["text"]
        data = data.fillna("NA").to_dict()

    return ResponseModel(status="success", data = data) 


@app.post("/elements/projection/compute", dependencies=[Depends(verified_user)])
async def compute_projection(project: Annotated[Project, Depends(get_project)],
                            username: Annotated[str, Header()],
                            projection:ProjectionModel) -> ResponseModel:
    """
    Start projection computation using futures
    Dedicated process, end with a file on the project
    projection__user.parquet
    TODO : intégrer directement dans la classe features ?
    """
    if len(projection.features) == 0:
        return ResponseModel(status="error", message="No feature")
    
    features = project.features.get(projection.features)
    args = {
            "features":features,
            "params":projection.params
            }

    if projection.method == "umap":
        try:
            e = UmapParams(**projection.params)
        except ValidationError as e:
            return ResponseModel(status="error", message=str(e))
        #future_result = server.executor.submit(functions.compute_umap, **args)
        unique_id = server.queue.add("projection", functions.compute_umap, args)
        project.features.projections[username] = {
                                                        "params":projection,
                                                        "method":"umap",
                                                        "queue":unique_id
                                                        #"future":future_result
                                                        }
        return ResponseModel(status = "waiting", message="Projection umap under computation")
    if projection.method == "tsne":
        try:
            e = TsneParams(**projection.params)
        except ValidationError as e:
            return ResponseModel(status="error", message=str(e))
        #future_result = server.executor.submit(functions.compute_tsne, **args)
        unique_id = server.queue.add("projection", functions.compute_tsne, args)
        project.features.projections[username] = {
                                                        "params":projection,
                                                        "method":"tsne",
                                                        "queue":unique_id
                                                        #"future":future_result
                                                        }
        return ResponseModel(status = "waiting", message="Projection tsne under computation")
    return ResponseModel(status="error", message="This projection is not available")


@app.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(project: Annotated[Project, Depends(get_project)],
                            scheme:str,
                            min:int = 0,
                            max:int = 0,
                            mode:str = "all",
                        ) -> ResponseModel:
    """
    Get table of elements
    """
    r = project.schemes.get_table(scheme, min, max, mode).fillna("NA")
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    return ResponseModel(status="success", data=r.to_dict())
    
@app.post("/elements/table", dependencies=[Depends(verified_user)])
async def post_list_elements(project: Annotated[Project, Depends(get_project)],
                            username: Annotated[str, Header()],
                            table:TableElementsModel
                            ) -> ResponseModel:
    """
    Post a table of annotations
    """
    r = project.schemes.push_table(table = table, 
                                   user = username, 
                                   action = table.action)
    server.log_action(username, "update data table", project.name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    return ResponseModel(status="success",  message=r["success"])

@app.post("/elements/zeroshot", dependencies=[Depends(verified_user)])
async def zeroshot(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    zshot:ZeroShotModel
                            ) -> ResponseModel:
    """
    Launch a call to an external API for 0-shot
    """
    # get subset of unlabelled elements
    df = project.schemes.get_table(zshot.scheme, 0, zshot.number, "untagged")
    # make the call
    r = await project.compute_zeroshot(df, zshot)
    if "error" in r:
        ResponseModel(status="error", message=r["error"])
    return ResponseModel(status="success", message="Annotation in progress")

@app.get("/elements/{element_id}", dependencies=[Depends(verified_user)])
async def get_element(project: Annotated[Project, Depends(get_project)],
                      username: Annotated[str, Header()],
                      element_id:str,
                      scheme:str) -> ResponseModel:
    """
    Get specific element
    TODO : remove exception
    """
    try:
        data = project.get_element(element_id, scheme=scheme, user=username)
        return ResponseModel(status="success", data=data)        
    except: # gérer la bonne erreur
        return ResponseModel(status="error", message=f"Element {element_id} not found")
    
@app.post("/tags/{action}", dependencies=[Depends(verified_user)])
async def post_tag(action:Action,
                   username: Annotated[str, Header()],
                   project: Annotated[Project, Depends(get_project)],
                   annotation:AnnotationModel) -> ResponseModel:
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
                                    username,
                                    annotation.selection
                                    )
        if "error" in r:
            return ResponseModel(status="error", message=r["error"])
        
        server.log_action(username, f"push annotation {annotation.element_id}", project.name)
        return ResponseModel(status="success", message="Label created")

    if action == "delete":
        r = project.schemes.delete_tag(annotation.element_id, 
                                   annotation.scheme,
                                   username
                                   ) # add user deletion
        if "error" in r:
            return ResponseModel(status="error", message=r["error"])
        
        server.log_action(username, f"delete annotation {annotation.element_id}", project.name)
        return ResponseModel(status="success", message="Label deleted")

# Schemes management
#-------------------

@app.get("/schemes", dependencies=[Depends(verified_user)])
async def get_schemes(project: Annotated[Project, Depends(get_project)],
                      scheme:str|None = None) -> ResponseModel:
    """
    Available scheme of a project
    """
    if scheme is None:
        data = project.schemes.get()
        return ResponseModel(status="success", data = data)
    
    a = project.schemes.available()
    if scheme in a:
        data = {"scheme":a[scheme]}
        return ResponseModel(status="success", data = data)
    
    return ResponseModel(status="error", message="scheme not available")


@app.post("/schemes/label/add", dependencies=[Depends(verified_user)])
async def add_label(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    scheme:str,
                    label:str,
                    ) -> ResponseModel:
    """
    Add a label to a scheme
    """
    r = project.schemes.add_label(label, scheme, username)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"add label {label} to {scheme}", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/schemes/label/delete", dependencies=[Depends(verified_user)])
async def delete_label(project: Annotated[Project, Depends(get_project)],
                       username: Annotated[str, Header()],
                    scheme:str,
                    label:str,
                    ) -> ResponseModel:
    """
    Remove a label from a scheme
    """
    r = project.schemes.delete_label(label, scheme, username)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"delete label {label} to {scheme}", project.name)
    return ResponseModel(status="success", message=r["success"])


@app.post("/schemes/{action}", dependencies=[Depends(verified_user)])
async def post_schemes(username: Annotated[str, Header()],
                        project: Annotated[Project, Depends(get_project)],
                        action:Action,
                        scheme:SchemeModel
                        ) -> ResponseModel:
    """
    Add, Update or Delete scheme
    TODO : user dans schememodel, necessary ?
    """
    if action == "add":
        r = project.schemes.add_scheme(scheme)
        if "error" in r:
            return ResponseModel(status="error", message=r["error"])
        server.log_action(username, f"add scheme {scheme.name}", project.name)
        return ResponseModel(status="success", message=r["success"])
    if action == "delete":
        r = project.schemes.delete_scheme(scheme)
        if "error" in r:
            return ResponseModel(status="error", message=r["error"])
        server.log_action(username, f"delete scheme {scheme.name}", project.name)
        return ResponseModel(status="success", message=r["success"])
    if action == "update":
        r = project.schemes.update_scheme(scheme.name, scheme.tags, username)
        if "error" in r:
            return ResponseModel(status="error", message=r["error"])
        server.log_action(username, f"update scheme {scheme.name}", project.name)
        return ResponseModel(status="success", message=r["success"])
    
    return ResponseModel(status="error", message="Wrong route")


# Features management
#--------------------

@app.get("/features", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]) -> ResponseModel:
    """
    Available scheme of a project
    """
    data = {"features":list(project.features.map.keys())}
    return ResponseModel(status="success", data=data)

@app.post("/features/add/regex", dependencies=[Depends(verified_user)])
async def post_regex(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    regex:RegexModel) -> ResponseModel:
    """
    Add a regex
    """
    r = project.add_regex(regex.name,regex.value)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"add regex {regex.name}", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/features/add/{name}", dependencies=[Depends(verified_user)])
async def post_embeddings(project: Annotated[Project, Depends(get_project)],
                          username: Annotated[str, Header()],
                          name:str,
                          params:ParamsModel
                          ) -> ResponseModel:
    """
    Compute features :
    common logic : call a function in a specific process, 
    create a file, and then updated
    TODO : refactorize the function + merge with regex
    """
    if name in project.features.training:
        return ResponseModel(status="error", message = "This feature is already in training")
    if not name in {"sbert","fasttext","dfm"}:
        return ResponseModel(status="error", message = "Not implemented")
    print("compute", name)
    df = project.content[project.params.col_text]
    if name == "sbert":
        args = {
                "texts":df,
                "model":"distiluse-base-multilingual-cased-v1"
                }
        func = functions.to_sbert
    if name == "fasttext":
        args = {
                "texts":df,
                "language":project.params.language,
                "path_models":server.path_models
                }
        func = functions.to_fasttext    
    if name == "dfm":
        # TODO save params with list to dict
        args = params.params
        args["texts"] = df
        func = functions.to_dtm
    #future_result = server.executor.submit(func, **args)
    unique_id = server.queue.add("feature", func, args)
    #project.features.training[name] = future_result
    project.features.training[name] = unique_id
    server.log_action(username, f"Compute feature dfm", project.name)
    return ResponseModel(status="success", message=f"computing {name}, it could take a few minutes")

@app.post("/features/delete/{name}", dependencies=[Depends(verified_user)])
async def delete_feature(project: Annotated[Project, Depends(get_project)],
                        username: Annotated[str, Header()],
                        name:str) -> ResponseModel:
    """
    Delete a specific feature
    """
    r = project.features.delete(name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"delete feature {name}", project.name)
    return ResponseModel(status="success", message=r["success"])

# Models management
#------------------

@app.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(project: Annotated[Project, Depends(get_project)]) -> ResponseModel:
    """
    Simplemodel parameters
    """
    data = project.simplemodels.available()
    return ResponseModel(status="success", data=data)

@app.post("/models/simplemodel", dependencies=[Depends(verified_user)])
async def post_simplemodel(project: Annotated[Project, Depends(get_project)],
                           username: Annotated[str, Header()],
                           simplemodel:SimpleModelModel) -> ResponseModel:
    """
    Compute simplemodel
    TODO : user out of simplemodel
    """
    r = project.update_simplemodel(simplemodel)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    return ResponseModel(status="success", message=r["success"])

@app.get("/models/bert", dependencies=[Depends(verified_user)])
async def get_bert(project: Annotated[Project, Depends(get_project)],
                   name:str)  -> ResponseModel: 
    """
    Bert parameters and statistics
    """
    b = project.bertmodels.get(name, lazy= True)
    if b is None:
        return ResponseModel(status="error", message="Bert model does not exist")
    data =  b.informations()
    return ResponseModel(status="success", data = data)

@app.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict(project: Annotated[Project, Depends(get_project)],
                  username: Annotated[str, Header()],
                     model_name:str,
                     data:str = "all")  -> ResponseModel:
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
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"predict bert {model_name}", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    bert:BertModelModel
                    )-> ResponseModel:
    """ 
    Compute bertmodel
    TODO : gestion du nom du projet/scheme à la base du modèle
    TODO : test if bert.params is well formed, maybe with pydantic ?
    """
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
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"train bert {bert.name}", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                     ) -> ResponseModel:
    """
    Stop user process
    """
    r = project.bertmodels.stop_user_process(username)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"stop bert training", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/models/bert/test", dependencies=[Depends(verified_user)])
async def start_test(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    scheme: str,
                    model:str
                     ) -> ResponseModel:
    """
    Start testing the model on the test set
    TODO : get scheme from bert model name
    """
    if project.schemes.test is None:
        return ResponseModel(status="error", message="No test dataset for this project")

    # get data labels + text
    df = project.schemes.get_scheme_data(scheme, complete=True, kind=["test"])

    if len(df["labels"].dropna())<10:
        return ResponseModel(status="error", message="Less than 10 elements annotated")

    # launch testing process : prediction
    r = project.bertmodels.start_testing_process(name = model,
                                                 user = username,
                                                df = df,
                                                col_text = "text",
                                                col_labels="labels",
                                                )
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"predict bert for testing", project.name)
    return ResponseModel(status="success", message="Bert prediction for final test launched")

@app.post("/models/bert/delete", dependencies=[Depends(verified_user)])
async def delete_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                    bert_name:str) -> ResponseModel:
    """
    Delete trained bert model
    """
    r = project.bertmodels.delete(bert_name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"delete bert model {bert_name}", project.name)
    return ResponseModel(status="success", message=r["success"])

@app.post("/models/bert/rename", dependencies=[Depends(verified_user)])
async def save_bert(project: Annotated[Project, Depends(get_project)],
                    username: Annotated[str, Header()],
                     former_name:str,
                     new_name:str) -> ResponseModel:
    """
    Rename bertmodel
    """
    r = project.bertmodels.rename(former_name, new_name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    server.log_action(username, f"rename bert model {former_name} - {new_name}", project.name)
    return ResponseModel(status="success", message=r["success"])


# Export elements
#----------------

@app.get("/export/data", dependencies=[Depends(verified_user)])
async def export_data(project: Annotated[Project, Depends(get_project)],
                      scheme:str,
                      format:str) -> FileResponse:
    """
    Export labelled data
    """
    r = project.export_data(format=format, scheme=scheme)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    r = FileResponse(r["path"], filename=r["name"])
    return r

@app.get("/export/features", dependencies=[Depends(verified_user)])
async def export_features(project: Annotated[Project, Depends(get_project)],
                          features:list = Query(),
                          format:str = Query()) -> FileResponse:
    """
    Export features
    """
    r = project.export_features(features = features, format=format)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    r = FileResponse(r["path"], filename=r["name"])
    return r

@app.get("/export/prediction", dependencies=[Depends(verified_user)])
async def export_prediction(project: Annotated[Project, Depends(get_project)],
                          format:str = Query(),
                          name:str = Query()) -> FileResponse:
    """
    Export annotations
    """
    r = project.bertmodels.export_prediction(name = name, format=format)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    r = FileResponse(r["path"], filename=r["name"])
    return r

@app.get("/export/bert", dependencies=[Depends(verified_user)])
async def export_bert(project: Annotated[Project, Depends(get_project)],
                          name:str = Query()) -> FileResponse:
    """
    Export fine-tuned BERT model
    """
    r = project.bertmodels.export_bert(name = name)
    if "error" in r:
        return ResponseModel(status="error", message=r["error"])
    r = FileResponse(r["path"], filename=r["name"])
    return r