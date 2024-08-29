import time
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Header,
    UploadFile,
    File,
    Query,
    Form,
    Request,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Annotated, List, Dict, Any
from jose import JWTError
import importlib
from pydantic import ValidationError
from activetigger.server import Server, Project
import activetigger.functions as functions
from activetigger.datamodels import (
    ProjectModel,
    ProjectDataModel,
    ProjectionInStrictModel,
    TableInModel,
    TableOutModel,
    ActionModel,
    AnnotationModel,
    SchemeModel,
    ProjectionInModel,
    ProjectionOutModel,
    TokenModel,
    SimpleModelModel,
    BertModelModel,
    FeatureModel,
    UmapModel,
    TsneModel,
    NextInModel,
    ElementOutModel,
    ZeroShotModel,
    UserInDBModel,
    UserModel,
    UsersServerModel,
    ProjectStateModel,
    QueueModel,
    ProjectDescriptionModel,
    ProjectAuthsModel,
    WaitingModel,
    DocumentationModel,
    TableLogsModel,
    ReconciliationModel,
    AuthActions,
    AvailableProjectsModel,
    TableAnnotationsModel,
)


# General comments
# - all post are logged
# - header identification with token
# - username is in the header


def test_rights(action: str, username: str, project_slug: str | None = None) -> bool:
    """
    Management of rights on the routes
    Different levels:
    - create project (only user status)
    - modify user (only user status)
    - modify project (user - project)
    - modify project element (user - project)
    Based on:
    - status of the account
    - relation to the project
    """

    user = server.users.get_user(name=username)
    status = user.status

    # possibility to create project
    if action == "create project":
        if status in ["root", "manager"]:
            return True
        else:
            raise HTTPException(403, "No rights for this action")

    # possibility to create user
    if action == "modify user":
        if status in ["root", "manager"]:
            return True
        else:
            raise HTTPException(403, "No rights for this action")

    if not project_slug:
        raise HTTPException(500, "Project name missing")

    auth = server.users.auth(username, project_slug)
    # print(auth)

    # possibility to modify project (create/delete)
    if action == "modify project":
        if (auth == "manager") or (status == "root"):
            return True
        else:
            raise HTTPException(403, "No rights for this action")

    # possibility to create elements of a project
    if action == "modify project element":
        if (auth == "manager") or (status == "root"):
            return True
        else:
            raise HTTPException(403, "No rights for this action")
    raise HTTPException(404, "No action found")


#######
# API #
#######

# to log specific events from api
logging.basicConfig(
    filename="log_server.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api")

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


app = FastAPI(lifespan=lifespan)  # defining the fastapi app
app.mount("/static", StaticFiles(directory=server.path / "static"), name="static")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token"
)  # defining the authentification object


async def check_processes(timer, step: int = 1) -> None:
    """
    Function called to update app state
    (i.e. joining parallel processes)
    Limited to once per time interval
    """
    # max one update per second to avoid excessive action
    if (time.time() - timer) < step:
        return None
    timer = time.time()

    # check the queue to see if process are completed
    server.queue.check()

    # update processes for active projects
    to_del = []
    for p, project in server.projects.items():
        # if project existing since one day, remove it from memory
        if (timer - project.starting_time) > 86400:
            to_del.append(p)

        # update pending processes
        project.features.update_processes()
        project.simplemodels.update_processes()
        predictions = project.bertmodels.update_processes()

        # if predictions completed, add them as features
        # careful : they are categorical variables
        if len(predictions) > 0:
            for f in predictions:
                df_num = functions.cat2num(predictions[f])
                print(df_num)
                name = f.replace("__", "_")
                project.features.add(name, df_num)  # avoid __ in the name for features
                print("Add feature", name)

    # delete old project (they will be loaded if needed)
    for p in to_del:
        del server.projects[p]


@app.middleware("http")
async def middleware(request: Request, call_next):
    """
    Middleware to take care of completed processes
    Executed at each action on the server
    """
    await check_processes(timer)
    response = await call_next(request)
    return response


app.add_middleware(
    CORSMiddleware,
    # TODO: move origins in config
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------
# Dependencies
# ------------


async def get_project(project_slug: str) -> ProjectModel:
    """
    Dependencie to get existing project
    - if already loaded, return it
    - if not loaded, load it first
    """

    # if project doesn't exist
    if not server.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project not found")

    # if the project is already loaded
    if project_slug in server.projects:
        return server.projects[project_slug]

    # load it
    server.start_project(project_slug)
    return server.projects[project_slug]


async def verified_user(
    request: Request, token: Annotated[str, Depends(oauth2_scheme)]
) -> UserInDBModel:
    """
    Dependency to test if the user is authentified with its token
    """
    # decode token
    try:
        payload = server.decode_access_token(token)
        if "error" in payload:
            raise HTTPException(status_code=403, detail=payload["error"])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Problem with token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Problem with token")

    # authentification
    user = server.users.get_user(name=username)

    if "error" in user:
        raise HTTPException(status_code=404, detail=user["error"])

    return user


async def check_auth_exists(
    request: Request,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str | None = None,
) -> None:
    """
    Check if a user is associated to a project
    """
    # print("route", request.url.path, request.method)
    auth = server.users.auth(current_user.username, project_slug)
    if not auth:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid rights")
    return None


async def check_auth_manager(
    request: Request,
    username: Annotated[str, Header()],
    project_slug: str | None = None,
) -> None:
    """
    Check if a user has auth to a project
    """
    if username == "root":  # root have complete power
        return None
    auth = server.users.auth(username, project_slug)
    if not auth == "manager":
        raise HTTPException(status_code=403, detail="Forbidden: Invalid rights")
    return None


# ------
# Routes
# ------


@app.get("/", response_class=HTMLResponse)
async def welcome() -> str:
    """
    Welcome page at the root path for the API
    """
    data_path = importlib.resources.files("activetigger")
    with open(data_path / "html/welcome.html", "r") as f:
        r = f.read()
    return r


@app.get("/documentation")
async def get_documentation() -> DocumentationModel:
    """
    Path for documentation
    Comments:
        For the moment, a dictionnary
    """
    data = {
        "credits": ["Julien Boelaert", "Étienne Ollion", "Émilien Schultz"],
        "contact": "emilien.schultz@ensae.fr",
        "page": "https://github.com/emilienschultz/pyactivetigger",
        "documentation": "To write ....",
    }
    return DocumentationModel(**data)


# Users
# ------


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()]
) -> TokenModel:
    """
    Authentificate user and return token
    """
    user = server.users.authenticate_user(form_data.username, form_data.password)
    if not isinstance(user, UserInDBModel):
        raise HTTPException(status_code=401, detail=user["error"])
    access_token = server.create_access_token(
        data={"sub": user.username}, expires_min=60
    )
    return TokenModel(
        access_token=access_token, token_type="bearer", status=user.status
    )


@app.post("/users/disconnect", dependencies=[Depends(verified_user)])
async def disconnect_user(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
    """
    Revoke user connexion
    """
    server.revoke_access_token(token)
    return None


@app.get("/users/me")
async def read_users_me(
    current_user: Annotated[UserInDBModel, Depends(verified_user)]
) -> UserModel:
    """
    Information on current user
    """
    return UserModel(username=current_user.username, status=current_user.status)


@app.get("/users", dependencies=[Depends(verified_user)])
async def existing_users() -> UsersServerModel:
    """
    Get existing users
    """
    return UsersServerModel(
        users=server.users.existing_users(), auth=["manager", "annotator"]
    )


@app.post("/users/create", dependencies=[Depends(verified_user)])
async def create_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    username_to_create: str = Query(),
    password: str = Query(),
    status: str = Query(),
) -> None:
    """
    Create user
    """
    test_rights("modify user", current_user.username)
    r = server.users.add_user(
        username_to_create, password, status, current_user.username
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return None


@app.post("/users/delete", dependencies=[Depends(verified_user)])
async def delete_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], user_to_delete: str
) -> None:
    """
    Delete user
    """
    test_rights("modify user", current_user.username)
    r = server.users.delete_user(user_to_delete)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return None


@app.post("/users/auth/{action}", dependencies=[Depends(verified_user)])
async def set_auth(
    action: AuthActions,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    username: str = Query(),
    project_slug: str = Query(),
    status: str = Query(None),
) -> None:
    """
    Set user auth
    """
    test_rights("modify project", current_user.username, project_slug)
    if action == "add":
        if not status:
            raise HTTPException(status_code=400, detail="Missing status")
        r = server.users.set_auth(username, project_slug, status)
        return None

    if action == "delete":
        # prevent to destroy root auth to access projects
        if username == "root":
            raise HTTPException(status_code=403, detail="Forbidden to delete root auth")
        r = server.users.delete_auth(username, project_slug)
        return None

    raise HTTPException(status_code=400, detail="Action not found")


@app.get("/users/auth", dependencies=[Depends(verified_user)])
async def get_auth(username: str) -> List:
    """
    Get all user auth
    """
    return server.users.get_auth(username, "all")


@app.get("/logs", dependencies=[Depends(verified_user)])
async def get_logs(
    username: str, project_slug: str = "all", limit=100
) -> TableLogsModel:
    """
    Get all logs for a username/project
    """
    df = server.get_logs(username, project_slug, limit)
    return TableLogsModel(
        time=list(df["time"]),
        user=list(df["user"]),
        project=list(df["project"]),
        action=list(df["action"]),
    )


# Projects management
# --------------------


@app.get(
    "/projects/{project_slug}",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
async def get_project_state(
    project: Annotated[Project, Depends(get_project)]
) -> ProjectStateModel:
    """
    Get the state of a specific project
    """
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    data = project.get_state()
    return ProjectStateModel(**data)


@app.get("/projects/{project_slug}/statistics", dependencies=[Depends(verified_user)])
async def get_project_statistics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None = None,
) -> ProjectDescriptionModel:
    """
    Statistics for a scheme and a user
    """
    r = project.get_description(scheme=scheme, user=current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ProjectDescriptionModel(**r)


@app.get("/projects")
async def get_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)]
) -> AvailableProjectsModel:
    """
    Get general informations on the server
    depending of the status of connected user
    """
    r = server.get_projects(current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return AvailableProjectsModel(projects=r)


@app.get("/queue")
async def get_queue() -> dict:
    """
    Get the state of the server queue
    """
    r = server.queue.state()
    print(r)
    print("COUCOU")
    return r
    # return QueueModel(**r)


@app.get("/queue/num")
async def get_nb_queue() -> int:
    """
    Get the number of element active in the queue
    """
    return server.queue.get_nb_active_processes()


@app.get("/projects/description", dependencies=[Depends(verified_user)])
async def get_description(
    project: Annotated[Project, Depends(get_project)],
    scheme: str | None = None,
    user: str | None = None,
) -> ProjectDescriptionModel:
    """
    Description of a specific element
    """
    r = project.get_description(scheme=scheme, user=user)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    print(r)
    return ProjectDescriptionModel(**r)


@app.get("/auth/project", dependencies=[Depends(verified_user)])
async def get_project_auth(project_slug: str) -> ProjectAuthsModel:
    """
    Users auth on a project
    """
    if not server.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project doesn't exist")
    r = server.users.get_project_auth(project_slug)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ProjectAuthsModel(auth=r)


@app.post("/projects/testdata", dependencies=[Depends(verified_user)])
async def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: Annotated[UploadFile, File()],
    col_text: str = Form(),
    col_id: str = Form(),
    n_test: int = Form(),
) -> None:
    """
    Add a dataset for test
    """
    r = project.add_testdata(file, col_text, col_id, n_test)

    # log action
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    # if success, update also parameters of the project
    server.set_project_parameters(project.params, current_user.username)

    # log action
    server.log_action(current_user.username, "add testdata project", project.name)

    return None


@app.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: ProjectDataModel,
) -> None:
    """
    Load new project
    """
    # test rights to create project
    test_rights("create project", current_user.username)

    # test if project_slug already exists
    if server.exists(project.project_name):
        raise HTTPException(
            status_code=500, detail="Project name already exists (exact or slugified)"
        )

    # create the project
    server.create_project(project, current_user.username)

    # log action
    server.log_action(current_user.username, "create project", project.project_name)

    return None


@app.post(
    "/projects/delete",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
async def delete_project(
    project_slug: str,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Delete a project
    """
    test_rights("modify project", current_user.username, project_slug)

    r = server.delete_project(project_slug)
    server.log_action(current_user.username, "delete project", project_slug)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return None


@app.post("/elements/next", dependencies=[Depends(verified_user)])
async def get_next(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    next: NextInModel,
) -> ElementOutModel:
    """
    Get next element
    """
    print(next)
    r = project.get_next(
        scheme=next.scheme,
        selection=next.selection,
        sample=next.sample,
        user=current_user.username,
        tag=next.tag,
        history=next.history,
        frame=next.frame,
        filter=next.filter,
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ElementOutModel(**r)


@app.get("/elements/projection", dependencies=[Depends(verified_user)])
async def get_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None,
) -> ProjectionOutModel | None:
    """
    Get projection data if computed
    """
    if not current_user.username in project.features.projections:
        return None

    if scheme is None:  # only the projection without specific annotations
        data = (
            project.features.projections[current_user.username]["data"]
            .fillna("NA")
            .to_dict()
        )
    else:  # add existing annotations in the data
        data = project.features.projections[current_user.username]["data"]
        df = project.schemes.get_scheme_data(scheme, complete=True)
        data["labels"] = df["labels"]
        data["texts"] = df["text"]
        data = data.fillna("NA")
        return ProjectionOutModel(
            index=list(data.index),
            x=list(data[0]),
            y=list(data[1]),
            labels=list(data["labels"]),
            texts=list(data["texts"]),
            status=project.features.projections[current_user.username]["id"],
        )
    raise HTTPException(status_code=400, detail="Projection problem")


@app.get("/elements/projection/current", dependencies=[Depends(verified_user)])
async def get_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None,
) -> ProjectionOutModel | WaitingModel:
    """
    Get projection data if computed
    """
    if not current_user.username in project.features.projections:
        raise HTTPException(
            status_code=404,
            detail="There is no projection available or under computation",
        )

    if not "data" in project.features.projections[current_user.username]:
        return WaitingModel(status="computing", detail="Computing projection")

    if scheme is None:  # only the projection without specific annotations
        data = (
            project.features.projections[current_user.username]["data"]
            .fillna("NA")
            .to_dict()
        )
    else:  # add existing annotations in the data
        data = project.features.projections[current_user.username]["data"]
        df = project.schemes.get_scheme_data(scheme, complete=True)
        data["labels"] = df["labels"]
        data["texts"] = df["text"]
        data = data.fillna("NA")
        print(data)
        return ProjectionOutModel(
            index=list(data.index),
            x=list(data[0]),
            y=list(data[1]),
            labels=list(data["labels"]),
            texts=list(data["texts"]),
            status="computed",
        )

    raise HTTPException(status_code=400, detail="No computation possible")


@app.post("/elements/projection/compute", dependencies=[Depends(verified_user)])
async def compute_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    projection: ProjectionInStrictModel,
) -> WaitingModel:
    """
    Start projection computation using futures
    Dedicated process, end with a file on the project
    projection__user.parquet
    """
    print(projection)
    if len(projection.features) == 0:
        raise HTTPException(status_code=400, detail="No feature available")

    features = project.features.get(projection.features)

    if projection.method == "umap":
        try:
            # e = UmapModel(**projection.params)
            e = UmapModel(**projection.params.__dict__)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail=str(e))

        args = {"features": features, "params": e.__dict__}
        unique_id = server.queue.add("projection", functions.compute_umap, args)
        project.features.projections[current_user.username] = {
            "params": projection,
            "method": "umap",
            "queue": unique_id,
        }
        return WaitingModel(detail="Projection umap under computation")

    if projection.method == "tsne":
        try:
            # e = TsneModel(**projection.params)
            e = TsneModel(**projection.params.__dict__)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail=str(e))
        args = {"features": features, "params": e.__dict__}
        unique_id = server.queue.add("projection", functions.compute_tsne, args)
        project.features.projections[current_user.username] = {
            "params": projection,
            "method": "tsne",
            "queue": unique_id,
        }
        return WaitingModel(detail="Projection tsne under computation")
    raise HTTPException(status_code=400, detail="Projection not available")


@app.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
    min: int = 0,
    max: int = 0,
    contains: str | None = None,
    mode: str = "all",
) -> TableOutModel:
    """
    Get table of elements
    """
    df = project.schemes.get_table(scheme, min, max, mode, contains).fillna("NA")
    if "error" in df:
        raise HTTPException(status_code=500, detail=df["error"])
    table = (df[["index", "timestamp", "labels", "text"]]).to_dict(orient="records")
    return TableOutModel(
        items=table,
        total=project.schemes.get_total(),
    )


# @app.post("/annotation/table", dependencies=[Depends(verified_user)])
# async def post_list_elements(
#     project: Annotated[Project, Depends(get_project)],
#     current_user: Annotated[UserInDBModel, Depends(verified_user)],
#     table: TableInModel,
# ) -> None:
#     """
#     Post a table of annotations
#     """
#     r = project.schemes.push_table(
#         table=table, user=current_user.username, action=table.action
#     )
#     server.log_action(current_user.username, "update data table", project.name)
#     if "error" in r:
#         raise HTTPException(status_code=500, detail=r["error"])
#     return None


@app.post("/annotation/table", dependencies=[Depends(verified_user)])
async def post_list_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    table: TableAnnotationsModel,
) -> None:
    """
    Update a table of annotations
    """
    print(table)
    errors = []
    # loop on annotations
    for annotation in table.annotations:
        if annotation.label is None or annotation.element_id is None:
            errors.append(annotation)
            continue

        r = project.schemes.push_tag(
            annotation.element_id,
            annotation.label,
            annotation.scheme,
            current_user.username,
            "add",
        )
        if "error" in r:
            errors.append(annotation)
            continue
        server.log_action(
            current_user.username,
            f"update annotation {annotation.element_id}",
            project.name,
        )

    if len(errors) > 0:
        raise HTTPException(
            status_code=500,
            detail="Error with some of the annotations - " + str(errors),
        )

    return None


@app.get("/elements/reconciliate", dependencies=[Depends(verified_user)])
async def get_reconciliation_table(
    project: Annotated[Project, Depends(get_project)], scheme: str
):
    """
    Get the reconciliation table
    """
    df = project.schemes.get_reconciliation_table(scheme)
    if "error" in df:
        raise HTTPException(status_code=500, detail=r["error"])
    return ReconciliationModel(list_disagreements=df.to_dict(orient="records"))


@app.post("/elements/reconciliate", dependencies=[Depends(verified_user)])
async def post_reconciliation(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    users: list = Query(),
    element_id: str = Query(),
    tag: str = Query(),
    scheme: str = Query(),
) -> None:
    """
    Post a label for all user in a list
    TODO : a specific action for reconciliation ?
    """

    # for each user
    for u in users:
        project.schemes.push_tag(element_id, tag, scheme, u, "add")
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

    # add a new tag for the reconciliator
    project.schemes.push_tag(
        element_id, tag, scheme, current_user.username, "reconciliation"
    )

    # log
    server.log_action(
        current_user.username,
        f"reconciliate annotation {element_id} for {users} with {tag}",
        project.name,
    )
    return None


@app.post("/elements/zeroshot", dependencies=[Depends(verified_user)])
async def zeroshot(
    project: Annotated[Project, Depends(get_project)],
    zshot: ZeroShotModel,
) -> WaitingModel:
    """
    Launch a call to an external API for 0-shot
    """
    # get subset of unlabelled elements
    df = project.schemes.get_table(zshot.scheme, 0, zshot.number, "untagged")
    # make the call
    r = await project.compute_zeroshot(df, zshot)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return WaitingModel(detail="Annotation in progress")


@app.get("/elements/{element_id}", dependencies=[Depends(verified_user)])
async def get_element(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    element_id: str,
    scheme: str,
) -> ElementOutModel:
    """
    Get specific element
    """
    r = project.get_element(element_id, scheme=scheme, user=current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ElementOutModel(**r)


@app.post("/annotation/{action}", dependencies=[Depends(verified_user)])
async def post_tag(
    action: ActionModel,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    annotation: AnnotationModel,
) -> None:
    """
    Add, Update, Delete annotations
    Comment :
    - For the moment add == update
    - No information kept of selection process
    """
    if action in ["add", "update"]:
        if annotation.label is None:
            raise HTTPException(status_code=422, detail="Missing a tag")
        r = project.schemes.push_tag(
            annotation.element_id,
            annotation.label,
            annotation.scheme,
            current_user.username,
            "add",
        )
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

        server.log_action(
            current_user.username,
            f"push annotation {annotation.element_id}",
            project.name,
        )
        return None

    if action == "delete":
        r = project.schemes.delete_tag(
            annotation.element_id, annotation.scheme, current_user.username
        )  # add user deletion
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

        server.log_action(
            current_user.username,
            f"delete annotation {annotation.element_id}",
            project.name,
        )
        return None

    raise HTTPException(status_code=400, detail="Wrong action")


@app.post("/stop", dependencies=[Depends(verified_user)])
async def stop_process(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop user process
    """
    if not current_user.username in project.bertmodels.computing:
        raise HTTPException(status_code=400, detail="Process missing")
    unique_id = project.bertmodels.computing[current_user.username][1]
    r = server.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, f"stop process", project.name)
    return None


# Schemes management
# -------------------


@app.post("/schemes/label/add", dependencies=[Depends(verified_user)])
async def add_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    label: str,
) -> None:
    """
    Add a label to a scheme
    """
    test_rights("modify project element", current_user.username, project.name)

    r = project.schemes.add_label(label, scheme, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    server.log_action(
        current_user.username, f"add label {label} to {scheme}", project.name
    )
    return None


@app.post("/schemes/label/delete", dependencies=[Depends(verified_user)])
async def delete_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    label: str,
) -> None:
    """
    Remove a label from a scheme
    """
    test_rights("modify project element", current_user.username, project.name)

    r = project.schemes.delete_label(label, scheme, current_user.username)
    if "error" in r:
        print(r["error"])
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(
        current_user.username, f"delete label {label} to {scheme}", project.name
    )
    return None


@app.post("/schemes/label/rename", dependencies=[Depends(verified_user)])
async def rename_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    former_label: str,
    new_label: str,
) -> None:
    """
    Add a label to a scheme
    - create new label (the order is important)
    - convert tags (need the label to exist, add a new element for each former)
    - delete former label
    """
    test_rights("modify project element", current_user.username, project.name)

    # test if the new label exist, either create it

    exists = project.schemes.exists_label(scheme, new_label)

    if not exists:
        r = project.schemes.add_label(new_label, scheme, current_user.username)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

    # convert the tags from the previous label
    r = project.schemes.convert_tags(
        former_label, new_label, scheme, current_user.username
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    # delete previous label in the scheme
    r = project.schemes.delete_label(former_label, scheme, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    # log
    server.log_action(
        current_user.username,
        f"rename label {former_label} to {new_label} in {scheme}",
        project.name,
    )
    return None


@app.post("/schemes/{action}", dependencies=[Depends(verified_user)])
async def post_schemes(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    action: ActionModel,
    scheme: SchemeModel,
) -> None:
    """
    Add, Update or Delete scheme
    """
    test_rights("modify project element", current_user.username, project.name)

    if action == "add":
        r = project.schemes.add_scheme(scheme)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"add scheme {scheme.name}", project.name
        )
        return None
    if action == "delete":
        r = project.schemes.delete_scheme(scheme, current_user.username)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"delete scheme {scheme.name}", project.name
        )
        return None
    if action == "update":
        r = project.schemes.update_scheme(
            scheme.name, scheme.tags, current_user.username
        )
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"update scheme {scheme.name}", project.name
        )
        return None
    raise HTTPException(status_code=400, detail="Wrong route")


# Features management
# --------------------


@app.get("/features", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]) -> List[str]:
    """
    Available scheme of a project
    """
    return list(project.features.map.keys())


@app.post("/features/add", dependencies=[Depends(verified_user)])
async def post_embeddings(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    feature: FeatureModel,
) -> WaitingModel | None:
    """
    Compute features :
    - same prcess
    - specific process : function + temporary file + update
    """
    test_rights("modify project", current_user.username, project.name)
    if feature.name in project.features.training:
        raise HTTPException(
            status_code=400, detail="This feature is already in training"
        )
    if not feature.type in {"sbert", "fasttext", "dfm", "regex"}:
        raise HTTPException(status_code=400, detail="Not implemented")

    # specific case of regex that is not parallelized yet
    if feature.type == "regex":
        if not "value" in feature.parameters:
            raise HTTPException(
                status_code=400, detail="Parameters missing for the regex"
            )
        regex_name = f"regex_[{feature.parameters['value']}]_by_{current_user.username}"
        regex_value = feature.parameters["value"]
        r = project.add_regex(regex_name, regex_value)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        server.log_action(
            f"add regex {regex_name}",
            project.name,
        )
        return None
    # case for computation on specific processes
    df = project.content["text"]
    if feature.type == "sbert":
        args = {"texts": df, "model": "distiluse-base-multilingual-cased-v1"}
        func = functions.to_sbert
    if feature.type == "fasttext":
        args = {
            "texts": df,
            "language": project.params.language,
            "path_models": server.path_models,
        }
        func = functions.to_fasttext
    if feature.type == "dfm":
        args = feature.parameters
        args["texts"] = df
        func = functions.to_dtm

    # add the computation to queue
    unique_id = server.queue.add("feature", func, args)
    project.features.training[feature.name] = unique_id

    server.log_action(current_user.username, f"Compute feature dfm", project.name)
    return WaitingModel(detail=f"computing {feature.type}, it could take a few minutes")


@app.post("/features/delete", dependencies=[Depends(verified_user)])
async def delete_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete a specific feature
    """
    test_rights("modify project", current_user.username, project.name)

    r = project.features.delete(name)
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    server.log_action(current_user.username, f"delete feature {name}", project.name)
    return None


# Models management
# ------------------


@app.post("/models/simplemodel", dependencies=[Depends(verified_user)])
async def post_simplemodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    simplemodel: SimpleModelModel,
) -> None:
    """
    Compute simplemodel
    """
    print(simplemodel)
    r = project.update_simplemodel(simplemodel, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return None


@app.get("/models/bert", dependencies=[Depends(verified_user)])
async def get_bert(
    project: Annotated[Project, Depends(get_project)], name: str
) -> Dict[str, Any]:
    """
    Get Bert parameters and statistics
    """
    b = project.bertmodels.get(name, lazy=True)
    if b is None:
        raise HTTPException(status_code=400, detail="Bert model does not exist")
    data = b.informations()
    return data


@app.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    data: str = "all",
) -> None:
    """
    Start prediction with a model
    """
    df = project.content[["text"]]  # get data

    # start process
    r = project.bertmodels.start_predicting_process(
        name=model_name, df=df, col_text="text", user=current_user.username
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, f"predict bert {model_name}", project.name)
    return None


@app.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    TODO : améliorer la gestion du nom du projet/scheme à la base du modèle
    """
    df = project.schemes.get_scheme_data(
        bert.scheme, complete=True
    )  # move it elswhere ?
    df = df[["text", "labels"]].dropna()  # remove non tag data
    r = project.bertmodels.start_training_process(
        name=bert.name,
        user=current_user.username,
        scheme=bert.scheme,
        df=df,
        col_text=df.columns[0],
        col_label=df.columns[1],
        base_model=bert.base_model,
        params=bert.params,
        test_size=bert.test_size,
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, f"train bert {bert.name}", project.name)
    return None


@app.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop user process
    """
    if not current_user.username in project.bertmodels.computing:
        raise HTTPException(status_code=400, detail="No process found")
    unique_id = project.bertmodels.computing[current_user.username][1]
    r = server.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, f"stop bert training", project.name)
    return None


@app.post("/models/bert/test", dependencies=[Depends(verified_user)])
async def start_test(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    model: str,
) -> None:
    """
    Start testing the model on the test set
    TODO : get scheme from bert model name
    """
    if project.schemes.test is None:
        raise HTTPException(status_code=500, detail="No test dataset for this project")

    # get data labels + text
    df = project.schemes.get_scheme_data(scheme, complete=True, kind=["test"])

    if len(df["labels"].dropna()) < 10:
        raise HTTPException(status_code=500, detail="Less than 10 elements annotated")

    # launch testing process : prediction
    r = project.bertmodels.start_testing_process(
        name=model,
        user=current_user.username,
        df=df,
        col_text="text",
        col_labels="labels",
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, f"predict bert for testing", project.name)
    return None


@app.post("/models/bert/delete", dependencies=[Depends(verified_user)])
async def delete_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert_name: str,
) -> None:
    """
    Delete trained bert model
    """
    test_rights("modify project", current_user.username, project.name)

    r = project.bertmodels.delete(bert_name)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(
        current_user.username, f"delete bert model {bert_name}", project.name
    )
    return None


@app.post("/models/bert/rename", dependencies=[Depends(verified_user)])
async def save_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename bertmodel
    """
    test_rights("modify project", current_user.username, project.name)

    r = project.bertmodels.rename(former_name, new_name)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(
        current_user.username,
        f"rename bert model {former_name} - {new_name}",
        project.name,
    )
    return None


# Export elements
# ----------------


@app.get("/export/data", dependencies=[Depends(verified_user)])
async def export_data(
    project: Annotated[Project, Depends(get_project)], scheme: str, format: str
) -> FileResponse:
    """
    Export labelled data
    """
    r = project.export_data(format=format, scheme=scheme)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


@app.get("/export/features", dependencies=[Depends(verified_user)])
async def export_features(
    project: Annotated[Project, Depends(get_project)],
    features: list = Query(),
    format: str = Query(),
) -> FileResponse:
    """
    Export features
    """
    r = project.export_features(features=features, format=format)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


@app.get("/export/prediction", dependencies=[Depends(verified_user)])
async def export_prediction(
    project: Annotated[Project, Depends(get_project)],
    format: str = Query(),
    name: str = Query(),
) -> FileResponse:
    """
    Export annotations
    """
    r = project.bertmodels.export_prediction(name=name, format=format)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


# @app.get("/export/bert", dependencies=[Depends(verified_user)])
# async def export_bert(
#     project: Annotated[Project, Depends(get_project)],
#     current_user: Annotated[UserInDBModel, Depends(verified_user)],
#     name: str = Query(),
# ) -> FileResponse:
#     """
#     Export fine-tuned BERT model
#     """
#     r = project.bertmodels.export_bert(name=name)
#     if "error" in r:
#         raise HTTPException(status_code=500, detail=r["error"])
#     return FileResponse(r["path"], filename=r["name"])


@app.get("/export/bert", dependencies=[Depends(verified_user)])
async def export_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> str:
    """
    Export fine-tuned BERT model
    """
    r = project.bertmodels.export_bert(name=name)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return "/static/" + r["name"]
