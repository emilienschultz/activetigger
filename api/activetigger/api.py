import importlib
import logging
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from datetime import datetime
from importlib.abc import Traversable
from io import StringIO
from typing import Annotated, Any, Callable

import pandas as pd
import psutil
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from jose import JWTError

from activetigger.datamodels import (
    ActionModel,
    AnnotationModel,
    AuthActions,
    AvailableProjectsModel,
    BertModelModel,
    CodebookModel,
    DocumentationModel,
    ElementOutModel,
    FeatureModel,
    GenerationCreationModel,
    GenerationModel,
    GenerationModelApi,
    GenerationRequest,
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
    UserGenerationComputing,
    UserInDBModel,
    UserModel,
    UserProjectionComputing,
    UsersServerModel,
    WaitingModel,
)
from activetigger.functions import get_gpu_memory_info
from activetigger.generation.generations import Generations
from activetigger.orchestrator import Orchestrator
from activetigger.project import Project

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
    try:
        user = orchestrator.users.get_user(name=username)
    except Exception as e:
        raise HTTPException(404) from e

    # general status
    status = user.status

    # TODO : check project auth

    # possibility to create project
    if action == "create project":
        if status in ["root", "manager"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to create user
    if action == "create user":
        if status in ["root"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to kill a process directly
    if action == "kill process":
        if status in ["root"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to modify user
    if action == "modify user":
        if status in ["root"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # get all information
    if action == "get all server information":
        if status == "root":
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    if not project_slug:
        raise HTTPException(500, "Project name missing")

    auth = orchestrator.users.auth(username, project_slug)
    # print(auth)

    # possibility to modify project (create/delete)
    if action == "modify project":
        if (auth == "manager") or (status == "root"):
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to create elements of a project
    if action == "modify project element":
        if (auth == "manager") or (status == "root"):
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to add/update annotations : everyone
    if action == "modify annotation":
        if (auth == "manager") or (status == "root"):
            return True
        elif auth == "annotator":
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # get project information
    if action == "get project information":
        if (auth == "manager") or (status == "root"):
            return True
        elif auth == "annotator":
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    raise HTTPException(404, "No action found")


#######
# API #
#######

# to log specific events from api
logger = logging.getLogger("api")
logger_simplemodel = logging.getLogger("simplemodel")

# starting the server
orchestrator = Orchestrator()
timer = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Frame the execution of the api
    """
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")
    orchestrator.queue.close()


# starting the app
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=orchestrator.path / "static"), name="static")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # defining the authentification object


async def check_processes(timer: float, step: int = 1) -> None:
    """
    Function to update server state
    (i.e. joining parallel processes)
    Limited to once per time interval
    """
    # max one update alllowed per step
    if (time.time() - timer) < step:
        return None

    # update processes for each active projects
    orchestrator.update()


@app.middleware("http")
async def middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]):
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
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project not found")

    # if the project is already loaded
    if project_slug in orchestrator.projects:
        return orchestrator.projects[project_slug]

    # Manage a FIFO queue when there is too many projects
    try:
        if len(orchestrator.projects) >= orchestrator.max_projects:
            old_element = sorted(
                [[p, orchestrator.projects[p].starting_time] for p in orchestrator.projects],
                key=lambda x: x[1],
            )[0]
            if (
                old_element[1] < time.time() - 3600
            ):  # check if the project has a least one hour old to avoid destroying current projects
                del orchestrator.projects[old_element[0]]
                print(f"Delete project {old_element[0]} to gain memory")
            else:
                print("Too many projects in the current memory")
                raise HTTPException(
                    status_code=500,
                    detail="There is too many projects currently loaded in this server. Please wait",
                )
    except Exception as e:
        print("PROBLEM IN THE FIFO QUEUE", e)

    # load the project
    orchestrator.start_project(project_slug)

    # return loaded project
    return orchestrator.projects[project_slug]


async def verified_user(
    request: Request, token: Annotated[str, Depends(oauth2_scheme)]
) -> UserInDBModel:
    """
    Dependency to test if the user is authentified with its token
    """
    # decode token
    try:
        payload = orchestrator.decode_access_token(token)
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Problem with token")
    except JWTError:
        raise HTTPException(status_code=401, detail="Problem with token")
    except Exception as e:
        raise HTTPException(status_code=403) from e

    # get user caracteristics
    try:
        user = orchestrator.users.get_user(name=username)
        return user
    except Exception as e:
        raise HTTPException(status_code=404) from e


async def check_auth_exists(
    request: Request,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> None:
    """
    Check if a user is associated to a project
    """
    # print("route", request.url.path, request.method)
    auth = orchestrator.users.auth(current_user.username, project_slug)
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
    data_path: Traversable = importlib.resources.files("activetigger")
    with open(data_path.joinpath("html", "welcome.html"), "r") as f:
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
    Authentificate user from username/password and return token
    """
    # authentificate the user
    try:
        user = orchestrator.users.authenticate_user(form_data.username, form_data.password)
    except Exception as e:
        raise HTTPException(status_code=401, detail="Wrong username or password") from e

    # create new token for the user
    access_token = orchestrator.create_access_token(data={"sub": user.username}, expires_min=120)
    return TokenModel(access_token=access_token, token_type="bearer", status=user.status)


@app.post("/users/disconnect", dependencies=[Depends(verified_user)])
async def disconnect_user(token: Annotated[str, Depends(oauth2_scheme)]) -> None:
    """
    Revoke user connexion
    """
    orchestrator.revoke_access_token(token)
    return None


@app.get("/users/me")
async def read_users_me(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UserModel:
    """
    Information on current user
    """
    return UserModel(username=current_user.username, status=current_user.status)


@app.get("/users")
async def existing_users(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> UsersServerModel:
    """
    Get existing users
    """
    users = orchestrator.users.existing_users()
    return UsersServerModel(
        users=users,
        auth=["manager", "annotator"],
    )


@app.get("/users/recent")
async def recent_users() -> list[str]:
    """
    Get recently connected users
    """
    users = orchestrator.db_manager.projects_service.get_current_users(300)
    return users


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
    test_rights("create user", current_user.username)
    try:
        orchestrator.users.add_user(
            username_to_create, password, status, current_user.username, mail
        )
    except Exception as e:
        raise HTTPException(status_code=500) from e
    return None


@app.post("/users/delete", dependencies=[Depends(verified_user)])
async def delete_user(
    current_user: Annotated[UserInDBModel, Depends(verified_user)], user_to_delete: str
) -> None:
    """
    Delete user
    - root can delete all
    - users can only delete account they created
    """
    # manage rights
    test_rights("modify user", current_user.username)
    try:
        orchestrator.users.delete_user(user_to_delete, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500) from e
    return None


@app.post("/users/changepwd", dependencies=[Depends(verified_user)])
async def change_password(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    pwdold: str = Query(),
    pwd1: str = Query(),
    pwd2: str = Query(),
):
    """
    Change password for an account
    """
    orchestrator.users.change_password(current_user.username, pwdold, pwd1, pwd2)
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
    Modify user auth on a specific project
    """
    test_rights("modify project", current_user.username, project_slug)
    if action == "add":
        if not status:
            raise HTTPException(status_code=400, detail="Missing status")
        try:
            orchestrator.users.set_auth(username, project_slug, status)
        except Exception as e:
            raise HTTPException(status_code=500) from e
        orchestrator.log_action(current_user.username, f"INFO add user {username}", "all")
        return None

    if action == "delete":
        try:
            orchestrator.users.delete_auth(username, project_slug)
        except Exception as e:
            raise HTTPException(status_code=500) from e
        orchestrator.log_action(current_user.username, f"INFO delete user {username}", "all")
        return None

    raise HTTPException(status_code=400, detail="Action not found")


@app.get("/users/auth", dependencies=[Depends(verified_user)])
async def get_auth(username: str) -> list:
    """
    Get all user auth
    """
    return orchestrator.users.get_auth(username, "all")


@app.get("/logs", dependencies=[Depends(verified_user)])
async def get_logs(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str = "all",
    limit: int = 100,
) -> TableOutModel:
    """
    Get all logs for a username/project
    """
    if project_slug == "all":
        test_rights("get all server information", current_user.username)
    else:
        test_rights("get project information", current_user.username, project_slug)
    df = orchestrator.get_logs(project_slug, limit)
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
    try:
        r = orchestrator.get_auth_projects(current_user.username)
        return AvailableProjectsModel(projects=r)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/server")
async def get_queue() -> dict:
    """
    Get the state of the server
    - queue
    - gpu use
    TODO : maybe add a buffer ?
    """

    active_projects = {}
    for p in orchestrator.projects:
        active_projects[p] = [
            {
                "unique_id": c.unique_id,
                "user": c.user,
                "kind": c.kind,
                "time": c.time,
            }
            for c in orchestrator.projects[p].computing
        ]

    # only running processes for the moment
    q = orchestrator.queue.state()
    queue = {i: q[i] for i in q if q[i]["state"] == "running"}

    gpu = get_gpu_memory_info()
    cpu = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()
    memory_info = psutil.virtual_memory()
    disk_info = psutil.disk_usage("/")

    r = {
        "queue": queue,
        "active_projects": active_projects,
        "gpu": gpu,
        "cpu": {"proportion": cpu, "total": cpu_count},
        "memory": {
            "proportion": memory_info.percent,
            "total": memory_info.total / (1024**3),
            "available": memory_info.available / (1024**3),
        },
        "disk": {
            "proportion": disk_info.percent,
            "total": disk_info.total / (1024**3),
        },
    }

    return r


@app.get("/queue/num")
async def get_nb_queue() -> int:
    """
    Get the number of element active in the server queue
    """
    return orchestrator.queue.get_nb_active_processes()


@app.post("/kill", dependencies=[Depends(verified_user)])
async def kill_process(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    unique_id: str,
) -> None:
    """
    Stop current generation
    """
    test_rights("kill process", current_user.username)
    r = orchestrator.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    orchestrator.log_action(current_user.username, f"kill process {unique_id}", "all")
    return None


@app.get("/auth/project", dependencies=[Depends(verified_user)])
async def get_project_auth(project_slug: str) -> ProjectAuthsModel:
    """
    Users auth on a project
    """
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project doesn't exist")
    try:
        r = orchestrator.users.get_project_auth(project_slug)
        return ProjectAuthsModel(auth=r)
    except Exception as e:
        raise HTTPException(status_code=500) from e


@app.post("/projects/testset", dependencies=[Depends(verified_user)])
async def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    testset: TestSetDataModel,
) -> None:
    """
    Add a dataset for test when there is none available
    """
    try:
        # add the data
        project.add_testdata(testset, current_user.username, project.name)
        # update parameters of the project
        orchestrator.set_project_parameters(project.params, current_user.username)
        # log action
        orchestrator.log_action(current_user.username, "INFO add testdata project", project.name)
        return None
    except Exception as e:
        raise HTTPException(status_code=500) from e


@app.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: ProjectDataModel,
) -> str:
    """
    Load new project
    """
    # test rights to create project
    test_rights("create project", current_user.username)

    try:
        # create the project
        r = orchestrator.create_project(project, current_user.username)
        # log action
        orchestrator.log_action(current_user.username, "INFO create project", project.project_name)
        return r["success"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    try:
        print("start delete")
        orchestrator.delete_project(project_slug)
        orchestrator.log_action(current_user.username, "INFO delete project", project_slug)
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        label=next.label,
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
    scheme: str,
) -> ProjectionOutModel | None:
    """
    Get projection data if computed
    """

    # check if a projection is available
    if current_user.username not in project.projections.available:
        return None

    # data not yet computed
    if "data" not in project.projections.available[current_user.username]:
        return None

    # create the data from projection and current scheme
    data = project.projections.available[current_user.username]["data"]
    df = project.schemes.get_scheme_data(scheme, complete=True)
    data["labels"] = df["labels"]
    data = data.fillna("NA")

    return ProjectionOutModel(
        index=list(data.index),
        x=list(data[0]),
        y=list(data[1]),
        labels=list(data["labels"]),
        # texts=list(data["texts"]),
        status=project.projections.available[current_user.username]["id"],
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
    # get features to project
    if len(projection.features) == 0:
        raise HTTPException(status_code=400, detail="No feature available")
    features = project.features.get(projection.features)

    # get func and validate parameters for projection
    try:
        r = project.projections.validate(projection.method, projection.params.__dict__)
    except Exception as e:
        raise HTTPException(status_code=500, detail=getattr(e, "message", repr(e))) from e

    # add to queue
    unique_id = orchestrator.queue.add(
        "projection",
        project.name,
        r["func"],
        {"features": features, "params": r["params"]},
    )
    if unique_id == "error":
        raise HTTPException(status_code=500, detail="Error in adding in the queue")
    project.computing.append(
        UserProjectionComputing(
            unique_id=unique_id,
            name=f"Projection by {current_user.username}",  # TODO: What to put here?
            user=current_user.username,
            time=datetime.now(),
            kind="projection",
            method=projection.method,
            params=projection,
        )
    )
    orchestrator.log_action(
        current_user.username,
        f"INFO compute projection {projection.method}",
        project.params.project_slug,
    )
    return WaitingModel(detail=f"Projection {projection.method} is computing")


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
    try:
        extract = project.schemes.get_table(scheme, min, max, mode, contains, dataset)
        df = extract.batch.fillna(" ")
        table = (df.reset_index()[["id", "timestamp", "labels", "text", "comment"]]).to_dict(
            orient="records"
        )
        return TableOutModel(
            items=table,
            total=extract.total,
        )
    except Exception as e:
        raise HTTPException(status_code=500) from e


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
        orchestrator.log_action(
            current_user.username,
            f"UPDATE ANNOTATION in {annotation.scheme}: {annotation.element_id} as {annotation.label}",
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
    orchestrator.log_action(
        current_user.username,
        f"RECONCILIATE ANNOTATION in {scheme}: {element_id} as {label}",
        project.name,
    )
    return None


@app.get("/elements/generate/models", dependencies=[Depends(verified_user)])
async def list_generation_models() -> list[GenerationModelApi]:
    """
    Returns the list of the available GenAI models for generation
    """
    return orchestrator.db_manager.generations_service.get_available_models()


@app.get("/elements/{project_slug}/generate/models", dependencies=[Depends(verified_user)])
async def list_project_generation_models(project_slug: str) -> list[GenerationModel]:
    """
    Returns the list of the available GenAI models configure for a project
    """
    return orchestrator.db_manager.generations_service.get_project_gen_models(project_slug)


@app.post("/elements/{project_slug}/generate/models", dependencies=[Depends(verified_user)])
async def add_project_generation_models(project_slug: str, model: GenerationCreationModel) -> int:
    """
    Add a new GenAI model for the project
    """
    return orchestrator.db_manager.generations_service.add_project_gen_model(project_slug, model)


@app.delete(
    "/elements/{project_slug}/generate/models/{model_id}",
    dependencies=[Depends(verified_user)],
)
async def delete_project_generation_models(project_slug: str, model_id: int) -> None:
    """
    Delete a GenAI model from the project
    """
    return orchestrator.db_manager.generations_service.delete_project_gen_model(
        project_slug, model_id
    )


@app.post("/elements/generate/start", dependencies=[Depends(verified_user)])
async def postgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenerationRequest,
) -> None:
    """
    Launch a call to generate from a prompt
    Only one possible by user
    """

    # get subset of unlabelled elements
    try:
        extract = project.schemes.get_table(request.scheme, 0, request.n_batch, request.mode)

    except Exception as e:
        raise HTTPException(status_code=500) from e

    model = orchestrator.db_manager.generations_service.get_gen_model(request.model_id)

    # create the independant process to manage the generation
    args = {
        "user": current_user.username,
        "project_name": project.name,
        "df": extract.batch,
        "prompt": request.prompt,
        "model": model,
    }

    unique_id = orchestrator.queue.add("generation", Generations.generate, args)


    if unique_id == "error":
        raise HTTPException(
            status_code=500, detail="Error in adding the generation call in the queue"
        )

    project.computing.append(
        UserGenerationComputing(
            unique_id=unique_id,
            user=current_user.username,
            project=project.name,
            model_id=request.model_id,
            number=request.n_batch,
            time=datetime.now(),
            kind="generation",
        )
    )

    orchestrator.log_action(
        current_user.username,
        "INFO Start generating process",
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
    p = project.get_process("generation", current_user.username)
    if len(p) == 0:
        raise HTTPException(status_code=400, detail="No process found for this user")
    unique_id = p[0].unique_id
    r = orchestrator.queue.kill(unique_id)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    orchestrator.log_action(
        current_user.username, "INFO stop generation", project.params.project_slug
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
        table = project.generations.get_generated(project.name, current_user.username, n_elements)
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
    r = project.get_element(element_id, scheme=scheme, user=current_user.username, dataset=dataset)
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

    # manage rights
    test_rights("modify annotation", current_user.username, project.name)

    if action in ["add", "update"]:
        try:
            r = project.schemes.push_annotation(
                annotation.element_id,
                annotation.label,
                annotation.scheme,
                current_user.username,
                annotation.dataset,
                annotation.comment,
            )

            orchestrator.log_action(
                current_user.username,
                f"ANNOTATE in {annotation.scheme}: tag {annotation.element_id} as {annotation.label} ({annotation.dataset})",
                project.name,
            )
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    if action == "delete":
        try:
            r = project.schemes.delete_annotation(
                annotation.element_id,
                annotation.scheme,
                annotation.dataset,
                current_user.username,
            )

            orchestrator.log_action(
                current_user.username,
                f"DELETE ANNOTATION in {annotation.scheme}: id {annotation.element_id}",
                project.name,
            )
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    r = project.schemes.convert_annotations(former_label, new_label, scheme, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    # delete previous label in the scheme
    r = project.schemes.delete_label(former_label, scheme, current_user.username)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])

    print("old label deleted")

    # log
    orchestrator.log_action(
        current_user.username,
        f"RENAME LABEL in {scheme}: label {former_label} to {new_label}",
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
        orchestrator.log_action(
            current_user.username, f"ADD LABEL in {scheme}: label {label}", project.name
        )
        return None

    if action == "delete":
        r = project.schemes.delete_label(label, scheme, current_user.username)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        orchestrator.log_action(
            current_user.username,
            f"DELETE LABEL in {scheme}: label {label}",
            project.name,
        )
        return None

    raise HTTPException(status_code=500, detail="Wrong action")


@app.post("/schemes/codebook", dependencies=[Depends(verified_user)])
async def post_codebook(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    codebook: CodebookModel,
) -> None:
    """
    Add codebook
    """
    test_rights("modify project element", current_user.username, project.name)

    r = project.schemes.add_codebook(codebook.scheme, codebook.content, codebook.time)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    orchestrator.log_action(
        current_user.username,
        f"MODIFY CODEBOOK: codebook {codebook.scheme}",
        project.name,
    )
    return None


@app.get("/schemes/codebook", dependencies=[Depends(verified_user)])
async def get_codebook(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
) -> CodebookModel:
    """
    Get the codebook of a scheme for a project
    """
    r = project.schemes.get_codebook(scheme)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return CodebookModel(
        scheme=scheme,
        content=str(r["codebook"]),
        time=str(r["time"]),
    )


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
        r = project.schemes.add_scheme(
            scheme.name, scheme.labels, scheme.kind, current_user.username
        )
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        orchestrator.log_action(
            current_user.username, f"ADD SCHEME: scheme {scheme.name}", project.name
        )
        return None
    if action == "delete":
        try:
            r = project.schemes.delete_scheme(scheme.name)
            orchestrator.log_action(
                current_user.username,
                f"DELETE SCHEME: scheme {scheme.name}",
                project.name,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        return None
    if action == "update":
        r = project.schemes.update_scheme(scheme.name, scheme.labels)
        if "error" in r:
            raise HTTPException(status_code=500, detail=r["error"])
        orchestrator.log_action(
            current_user.username, f"UPDATE SCHEME: scheme {scheme.name}", project.name
        )
        return None
    raise HTTPException(status_code=400, detail="Wrong route")


# Features management
# --------------------


@app.get("/features", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]) -> list[str]:
    """
    Available features for the project
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
    try:
        project.features.compute(
            df, feature.name, feature.type, feature.parameters, current_user.username
        )
        orchestrator.log_action(
            current_user.username, f"INFO Compute feature {feature.type}", project.name
        )
        return WaitingModel(
            detail=f"computing {feature.type}, it could take a few minutes"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    orchestrator.log_action(current_user.username, f"INFO delete feature {name}", project.name)
    return None


@app.get("/features/available", dependencies=[Depends(verified_user)])
async def get_feature_info(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict[str, Any]:
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
    orchestrator.log_action(current_user.username, "INFO compute simplemodel", project.name)
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
async def get_bert(project: Annotated[Project, Depends(get_project)], name: str) -> dict[str, Any]:
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
    batch_size: int = 32,
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
            raise Exception("Problem with full dataset")
    else:
        raise Exception(f"dataset {dataset} not found")

    # start process to predict
    r = project.bertmodels.start_predicting_process(
        name=model_name,
        user=current_user.username,
        df=df,
        col_text="text",
        dataset=dataset,
        batch_size=batch_size,
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    orchestrator.log_action(current_user.username, f"INFO predict bert {model_name}", project.name)
    return None


@app.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    TODO : move the methods to specific class
    """
    try:
        # get data
        df = project.schemes.get_scheme_data(bert.scheme, complete=True)
        df = df[["text", "labels"]].dropna()

        # management for multilabels / dichotomize
        if bert.dichotomize is not None:
            df["labels"] = df["labels"].apply(
                lambda x: project.schemes.dichotomize(x, bert.dichotomize)
            )
            bert.name = f"{bert.name}_multilabel_on_{bert.dichotomize}"

        # remove class under the threshold
        label_counts = df["labels"].value_counts()
        df = df[df["labels"].isin(label_counts[label_counts >= bert.class_min_freq].index)]

        # balance the dataset based on the min class
        if bert.class_balance:
            min_freq = df["labels"].value_counts().sort_values().min()
            df = (
                df.groupby("labels")
                .apply(lambda x: x.sample(min_freq))
                .reset_index(level=0, drop=True)
            )

        # launch training process
        project.bertmodels.start_training_process(
            name=bert.name,
            project=project.name,
            user=current_user.username,
            scheme=bert.scheme,
            df=df,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=bert.base_model,
            params=bert.params,
            test_size=bert.test_size,
        )
        orchestrator.log_action(current_user.username, f"INFO train bert {bert.name}", project.name)
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop user process
    """
    # get BERT process for username
    p = project.get_process("bert", current_user.username)
    if len(p) == 0:
        raise HTTPException(status_code=400, detail="No process found")
    # get id
    unique_id = p[0]["unique_id"]
    # kill the process
    r = orchestrator.queue.kill(unique_id)
    # delete it in the database
    project.bertmodels.projects_service.delete_model(project.name, p[0]["model"].name)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    orchestrator.log_action(current_user.username, "INFO stop bert training", project.name)
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
    orchestrator.log_action(current_user.username, "INFO predict bert for testing", project.name)
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
    orchestrator.log_action(
        current_user.username, f"INFO delete bert model {bert_name}", project.name
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
    orchestrator.log_action(
        current_user.username,
        f"INFO rename bert model {former_name} - {new_name}",
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
    try:
        r = project.export_features(features=features, format=format)
        return FileResponse(r["path"], filename=r["name"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/prediction/simplemodel", dependencies=[Depends(verified_user)])
async def export_simplemodel_predictions(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    format: str = "csv",
) -> FileResponse:
    """
    Export prediction simplemodel for the project/user/scheme if any
    """
    try:
        output, headers = project.simplemodels.export_prediction(
            scheme, current_user.username, format
        )
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    test_rights("modify project", current_user.username, project.name)
    try:
        r = project.bertmodels.export_bert(name=name)
        return "/static/" + r["name"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export/raw", dependencies=[Depends(verified_user)])
async def export_raw(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict:
    """
    Export raw data of the project
    """
    test_rights("modify project", current_user.username, project.name)
    try:
        return project.export_raw(project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        n_elements=str(number),
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
