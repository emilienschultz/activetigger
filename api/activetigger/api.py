import importlib
import logging
import time
from contextlib import asynccontextmanager
from io import StringIO
from typing import Annotated, Any, Dict, List

import pandas as pd
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError
from pydantic import ValidationError

import activetigger.functions as functions
from activetigger.datamodels import (
    ActionModel,
    AnnotationModel,
    AuthActions,
    AvailableProjectsModel,
    BertModelModel,
    DocumentationModel,
    ElementOutModel,
    FeatureModel,
    GenerateModel,
    NextInModel,
    ProjectAuthsModel,
    ProjectDataModel,
    ProjectDescriptionModel,
    ProjectionInStrictModel,
    ProjectionOutModel,
    ProjectModel,
    ProjectStateModel,
    ReconciliationModel,
    SchemeModel,
    SimpleModelModel,
    SimpleModelOutModel,
    TableAnnotationsModel,
    TableOutModel,
    TestSetDataModel,
    TokenModel,
    TsneModel,
    UmapModel,
    UserInDBModel,
    UserModel,
    UsersServerModel,
    WaitingModel,
)
from activetigger.server import Project, Server

# General comments
# - all post are logged
# - header identification with token
# - username is in the header


def test_rights(action: str, username: str, project_slug: str | None = None) -> bool:
    """
    Management of rights on the routes
    Different types of action:
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
logger = logging.getLogger("api")
logger_simplemodel = logging.getLogger("simplemodel")

# starting the server
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


# starting the app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=server.path / "static"), name="static")

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="token"
)  # defining the authentification object


async def check_processes(timer, step: int = 1) -> None:
    """
    Function to update server state
    (i.e. joining parallel processes)
    Limited to once per time interval
    """
    # max one update alllwoed per step
    if (time.time() - timer) < step:
        return None

    # update last update
    timer = time.time()

    # check the queue to see if process are completed
    server.queue.check()

    # update processes for each active projects
    to_del = []
    for p, project in server.projects.items():
        # if project existing since one day, remove it from memory
        if (timer - project.starting_time) > 86400:
            to_del.append(p)

        # update different pending processes
        project.features.update_processes()
        project.simplemodels.update_processes()
        project.generations.update_generations()
        predictions = project.bertmodels.update_processes()

        # if predictions completed, add them as features
        # careful : they are categorical variables
        if len(predictions) > 0:
            for f in predictions:
                df_num = functions.cat2num(predictions[f])
                name = f.replace("__", "_")
                project.features.add(
                    name=name,
                    kind="prediction",
                    parameters={},
                    username="system",
                    new_content=df_num,
                )  # avoid __ in the name for features
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


# allow multiple servers (avoir CORS error)
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
    Dependency to get existing project
    - if already loaded, return it
    - if not loaded, load it first
    """

    # if project doesn't exist
    if not server.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project not found")

    # if the project is already loaded
    if project_slug in server.projects:
        return server.projects[project_slug]

    # Manage a FIFO queue when there is too many projects
    if len(server.projects) >= server.max_projects:
        old_element = sorted(
            [[p, server.projects[p].starting_time] for p in server.projects],
            key=lambda x: x[1],
        )[0]
        if (
            old_element[1] < time.time() - 3600
        ):  # check if the project has a least one hour old to avoid destroying current projects
            del server.projects[old_element[0]]
            print(f"Delete project {old_element[0]} to gain memory")
        else:
            print("Too many projects in the current memory")
            raise HTTPException(
                status_code=500,
                detail="There is too many projects currently loaded in this server. Please wait",
            )

    # load the project
    server.start_project(project_slug)

    # return loaded project
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

    # get user caracteristics
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
    if not auth or "error" in auth:
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
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenModel:
    """
    Authentificate user from username/passwordand return token
    """
    # authentificate the user
    user = server.users.authenticate_user(form_data.username, form_data.password)

    # manage error
    if not isinstance(user, UserInDBModel):
        raise HTTPException(status_code=401, detail=user["error"])

    # create new token for the user
    access_token = server.create_access_token(
        data={"sub": user.username}, expires_min=120
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
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
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
    mail: str = Query(),
) -> None:
    """
    Create user
    """
    test_rights("modify user", current_user.username)
    r = server.users.add_user(
        username_to_create, password, status, current_user.username, mail
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
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        # log action
        server.log_action(current_user.username, f"add user {username}", "all")

        return None

    if action == "delete":
        # prevent to destroy root auth to access projects
        if username == "root":
            raise HTTPException(status_code=403, detail="Forbidden to delete root auth")
        r = server.users.delete_auth(username, project_slug)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        # log action
        server.log_action(current_user.username, f"delete user {username}", "all")
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
    username: str, project_slug: str = "all", limit: int = 100
) -> TableOutModel:
    """
    Get all logs for a username/project
    """
    df = server.get_logs(username, project_slug, limit)
    return TableOutModel(
        items=df.to_dict(orient="records"),
        total=limit,
    )


# Projects management
# --------------------


@app.get(
    "/projects/{project_slug}",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
async def get_project_state(
    project: Annotated[Project, Depends(get_project)],
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
    r = project.get_statistics(scheme=scheme, user=current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ProjectDescriptionModel(**r)


@app.get("/projects")
async def get_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> AvailableProjectsModel:
    """
    Get general informations on the server
    depending of the status of connected user
    """
    r = server.get_auth_projects(current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return AvailableProjectsModel(projects=r)


@app.get("/queue")
async def get_queue() -> dict:
    """
    Get the state of the server queue
    """
    r = server.queue.state()
    # only running processes for the moment
    return {i: r[i] for i in r if r[i]["state"] == "running"}


@app.get("/queue/num")
async def get_nb_queue() -> int:
    """
    Get the number of element active in the queue
    """
    return server.queue.get_nb_active_processes()


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


@app.post("/projects/testset", dependencies=[Depends(verified_user)])
async def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    testset: TestSetDataModel,
) -> None:
    """
    Add a dataset for test
    """
    r = project.add_testdata(testset)

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
    r = server.create_project(project, current_user.username)

    # raise error if needed
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

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
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    del server.projects[project_slug]
    server.log_action(current_user.username, "delete project", project_slug)
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
    r = project.get_next(
        scheme=next.scheme,
        selection=next.selection,
        sample=next.sample,
        user=current_user.username,
        label=next.tag,
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

    if scheme is None:
        raise HTTPException(status_code=400, detail="Please specify a scheme")

    # check if a projection is available
    if current_user.username not in project.features.projections:
        return None

    # data not yet computed
    if "data" not in project.features.projections[current_user.username]:
        return None

    # add existing annotations in the data
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
        if unique_id == "error":
            raise HTTPException(status_code=500, detail="Error in adding in the queue")
        project.features.projections[current_user.username] = {
            "params": projection,
            "method": "umap",
            "queue": unique_id,
        }
        server.log_action(
            current_user.username,
            "compute projection umap",
            project.params.project_slug,
        )
        return WaitingModel(detail="Projection umap is computing")

    if projection.method == "tsne":
        try:
            # e = TsneModel(**projection.params)
            e = TsneModel(**projection.params.__dict__)
        except ValidationError as e:
            raise HTTPException(status_code=500, detail=str(e))
        args = {"features": features, "params": e.__dict__}
        unique_id = server.queue.add("projection", functions.compute_tsne, args)
        if unique_id == "error":
            raise HTTPException(status_code=500, detail="Error in adding in the queue")
        project.features.projections[current_user.username] = {
            "params": projection,
            "method": "tsne",
            "queue": unique_id,
        }

        server.log_action(
            current_user.username,
            "compute projection tsne",
            project.params.project_slug,
        )
        return WaitingModel(detail="Projection tsne is computing")

    raise HTTPException(status_code=400, detail="Projection not available")


@app.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
    min: int = 0,
    max: int = 0,
    contains: str | None = None,
    mode: str = "all",
    dataset: str = "train",
) -> TableOutModel:
    """
    Get a table of elements
    """
    extract = project.schemes.get_table(scheme, min, max, mode, contains, dataset)
    if "error" in extract:
        raise HTTPException(status_code=500, detail=extract["error"])
    df = extract["batch"].fillna(" ")
    table = (df.reset_index()[["id", "timestamp", "labels", "text"]]).to_dict(
        orient="records"
    )
    return TableOutModel(
        items=table,
        total=extract["total"],
    )


@app.post("/annotation/table", dependencies=[Depends(verified_user)])
async def post_list_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    table: TableAnnotationsModel,
) -> None:
    """
    Update a table of annotations
    """
    errors = []
    # loop on annotations
    for annotation in table.annotations:
        if annotation.label is None or annotation.element_id is None:
            errors.append(annotation)
            continue

        r = project.schemes.push_annotation(
            annotation.element_id,
            annotation.label,
            annotation.scheme,
            current_user.username,
            table.dataset,
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
) -> ReconciliationModel:
    """
    Get the reconciliation table
    """
    try:
        df, users = project.schemes.get_reconciliation_table(scheme)
    except Exception:
        raise HTTPException(status_code=500, detail="Problem with the reconciliation")
    if "error" in df:
        raise HTTPException(status_code=500, detail=df["error"])
    return ReconciliationModel(table=df.to_dict(orient="records"), users=users)


@app.post("/elements/reconciliate", dependencies=[Depends(verified_user)])
async def post_reconciliation(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    users: list = Query(),
    element_id: str = Query(),
    label: str = Query(),
    scheme: str = Query(),
) -> None:
    """
    Post a label for all user in a list
    TODO : a specific action for reconciliation ?
    """

    # for each user
    for u in users:
        r = project.schemes.push_annotation(element_id, label, scheme, u, "train")
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

    # add a new tag for the reconciliator
    project.schemes.push_annotation(
        element_id, label, scheme, current_user.username, "reconciliation"
    )

    # log
    server.log_action(
        current_user.username,
        f"reconciliate annotation {element_id} for {users} with {label}",
        project.name,
    )
    return None


@app.post("/elements/generate/start", dependencies=[Depends(verified_user)])
async def postgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenerateModel,
) -> None:
    """
    Launch a call to generate from a prompt
    Only one possible by user
    """

    # get subset of unlabelled elements
    extract = project.schemes.get_table(
        request.scheme, 0, request.n_batch, request.mode
    )

    if "error" in extract:
        raise HTTPException(status_code=500, detail=extract["error"])

    # create the independant process to manage the generation
    args = {
        "user": current_user.username,
        "project_name": project.name,
        "df": extract["batch"],
        "api": request.api,
        "endpoint": request.endpoint,
        "prompt": request.prompt,
    }

    unique_id = server.queue.add("generation", functions.generate, args)

    if unique_id == "error":
        raise HTTPException(
            status_code=500, detail="Error in adding the generation call in the queue"
        )

    project.generations.generating[current_user.username] = {
        "unique_id": unique_id,
        "number": request.n_batch,
        "api": request.api,
    }

    server.log_action(
        current_user.username,
        "Start generating process",
        project.params.project_slug,
    )

    return None


@app.post("/elements/generate/stop", dependencies=[Depends(verified_user)])
async def stop_generation(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop current generation
    """
    if current_user.username not in project.generations.generating:
        raise HTTPException(status_code=400, detail="No process found for this user")
    unique_id = project.generations.generating[current_user.username]["unique_id"]
    r = server.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(
        current_user.username, "stop generation", project.params.project_slug
    )
    return None


@app.get("/elements/generate/elements", dependencies=[Depends(verified_user)])
async def getgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    n_elements: int,
) -> TableOutModel:
    """
    Get elements from prediction
    """
    try:
        table = project.generations.get_generated(
            project.name, current_user.username, n_elements
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error in loading generated data")

    # join with the text
    # table = table.join(project.content["text"], on="index")

    r = table.to_dict(orient="records")
    return TableOutModel(items=r, total=len(r))


@app.get("/elements/{element_id}", dependencies=[Depends(verified_user)])
async def get_element(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    element_id: str,
    scheme: str,
    dataset: str = "train",
) -> ElementOutModel:
    """
    Get specific element
    """
    r = project.get_element(
        element_id, scheme=scheme, user=current_user.username, dataset=dataset
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return ElementOutModel(**r)


@app.post("/annotation/{action}", dependencies=[Depends(verified_user)])
async def post_annotation(
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
        r = project.schemes.push_annotation(
            annotation.element_id,
            annotation.label,
            annotation.scheme,
            current_user.username,
            annotation.dataset,
            annotation.comment,
        )

        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

        server.log_action(
            current_user.username,
            f"push annotation {annotation.element_id} with the method {annotation.dataset}",
            project.name,
        )
        return None

    if action == "delete":
        r = project.schemes.delete_annotation(
            annotation.element_id,
            annotation.scheme,
            annotation.dataset,
            current_user.username,
        )
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])

        server.log_action(
            current_user.username,
            f"delete annotation {annotation.element_id}",
            project.name,
        )
        return None

    raise HTTPException(status_code=400, detail="Wrong action")


# Schemes management
# -------------------


@app.post("/schemes/label/rename", dependencies=[Depends(verified_user)])
async def rename_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    former_label: str,
    new_label: str,
) -> None:
    """
    Rename a a label
    - create new label (the order is important)
    - convert existing annotations (need the label to exist, add a new element for each former)
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
    r = project.schemes.convert_annotations(
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


@app.post("/schemes/label/{action}", dependencies=[Depends(verified_user)])
async def add_label(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    action: ActionModel,
    scheme: str,
    label: str,
) -> None:
    """
    Add a label to a scheme
    """
    test_rights("modify project element", current_user.username, project.name)

    if action == "add":
        r = project.schemes.add_label(label, scheme, current_user.username)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        server.log_action(
            current_user.username, f"add label {label} to {scheme}", project.name
        )
        return None

    if action == "delete":
        r = project.schemes.delete_label(label, scheme, current_user.username)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"delete label {label} to {scheme}", project.name
        )
        return None

    raise HTTPException(status_code=500, detail="Wrong action")


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
        r = project.schemes.add_scheme(scheme.name, scheme.tags)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"add scheme {scheme.name}", project.name
        )
        return None
    if action == "delete":
        r = project.schemes.delete_scheme(scheme.name)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        server.log_action(
            current_user.username, f"delete scheme {scheme.name}", project.name
        )
        return None
    if action == "update":
        r = project.schemes.update_scheme(scheme.name, scheme.tags)
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
    # manage rights
    test_rights("modify project", current_user.username, project.name)

    # get the data (for the moment, Features has no access to the data)
    df = project.content["text"]

    # compute the feature
    r = project.features.compute(
        df, feature.name, feature.type, feature.parameters, current_user.username
    )

    # manage error
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    # Log and return
    server.log_action(
        current_user.username, f"Compute feature {feature.type}", project.name
    )
    return WaitingModel(detail=f"computing {feature.type}, it could take a few minutes")


@app.post("/features/delete", dependencies=[Depends(verified_user)])
async def delete_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
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


@app.get("/features/available", dependencies=[Depends(verified_user)])
async def get_feature_info(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> Dict[str, Any]:
    """
    Get feature info
    """
    r = project.features.get_available()
    if "error" in r:
        raise HTTPException(status_code=400, detail=r["error"])
    return r


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
    r = project.update_simplemodel(simplemodel, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, "Compute simplemodel", project.name)
    logger_simplemodel.info("Start computing simplemodel")
    return None


@app.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
) -> SimpleModelOutModel | None:
    """
    Get available simplemodel for the project/user/scheme if any
    """
    r = project.simplemodels.get(scheme, current_user.username)
    if "error" in r:  # case where there is no model
        return None
    return SimpleModelOutModel(**r["success"])


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
    dataset: str = "all",
) -> None:
    """
    Start prediction with a model
    """
    # get the data
    if dataset == "train":
        df = project.content[["text"]]  # get data
    elif dataset == "all":
        r = project.features.get_column_raw("text", index="all")
        if "success" in r:
            df = pd.DataFrame(r["success"])
        else:
            return {"error": "Problem with full dataset"}
    else:
        return {"error": f"dataset {dataset} not found"}

    # start process to predict
    r = project.bertmodels.start_predicting_process(
        name=model_name,
        user=current_user.username,
        df=df,
        col_text="text",
        dataset=dataset,
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
    if current_user.username not in project.bertmodels.computing:
        raise HTTPException(status_code=400, detail="No process found")
    unique_id = project.bertmodels.computing[current_user.username][1]
    r = server.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    server.log_action(current_user.username, "stop bert training", project.name)
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
    server.log_action(current_user.username, "predict bert for testing", project.name)
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
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
    format: str,
    dataset: str = "train",
) -> FileResponse:
    """
    Export labelled data
    """
    r = project.export_data(format=format, scheme=scheme, dataset=dataset)
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
    r = project.bertmodels.export_prediction(
        name=name, file_name="predict_all.parquet", format=format
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


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


@app.get("/export/generations", dependencies=[Depends(verified_user)])
async def export_generations(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    number: int = Query(),
) -> FileResponse:
    """
    Export annotations
    """
    table = project.generations.get_generated(
        project_slug=project.name,
        username=current_user.username,
        n_elements=number,
    )

    if "error" in table:
        raise HTTPException(status_code=500, detail=table["error"])

    # join the text
    table = table.join(project.content["text"], on="index")

    # convert to payload
    output = StringIO()
    pd.DataFrame(table).to_csv(output, index=False)
    csv_data = output.getvalue()
    output.close()

    headers = {
        "Content-Disposition": 'attachment; filename="data.csv"',
        "Content-Type": "text/csv",
    }

    return Response(content=csv_data, media_type="text/csv", headers=headers)
