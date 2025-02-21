import importlib
import logging
import time
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from importlib.abc import Traversable
from typing import Annotated, Callable

import psutil
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    Response,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles

from activetigger import __version__
from activetigger.app.dependencies import (
    test_rights,
    verified_user,
)
from activetigger.app.routers import (
    annotations,
    export,
    features,
    generation,
    models,
    projects,
    schemes,
    users,
)
from activetigger.datamodels import (
    DocumentationModel,
    ProjectAuthsModel,
    TableOutModel,
    TokenModel,
    UserInDBModel,
)
from activetigger.functions import get_gpu_memory_info
from activetigger.orchestrator import orchestrator

# to log specific events from api
logger = logging.getLogger("api")
logger_simplemodel = logging.getLogger("simplemodel")

# starting the server
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

# add routers
app.include_router(users.router)
app.include_router(projects.router)
app.include_router(annotations.router)
app.include_router(schemes.router)
app.include_router(features.router)
app.include_router(export.router)
app.include_router(models.router)
app.include_router(generation.router)


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
async def middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
):
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


# --------------
# Generic routes
# --------------


@app.get("/", response_class=HTMLResponse)
async def welcome() -> str:
    """
    Welcome page at the root path for the API
    """
    data_path: Traversable = importlib.resources.files("activetigger")
    with open(str(data_path.joinpath("html", "welcome.html")), "r") as f:
        r = f.read()
    return r


@app.get("/documentation")
async def get_documentation() -> DocumentationModel:
    """
    Path for documentation
    Comments:
        For the moment, a dictionnary
    """
    return DocumentationModel(
        credits=["Julien Boelaert", "Étienne Ollion", "Émilien Schultz"],
        contact="emilien.schultz@ensae.fr",
        page="https://github.com/emilienschultz/pyactivetigger",
        documentation="To write ....",
    )


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenModel:
    """
    Authentificate user from username/password and return token
    """
    # authentificate the user
    try:
        user = orchestrator.users.authenticate_user(
            form_data.username, form_data.password
        )
    except Exception as e:
        raise HTTPException(status_code=401, detail="Wrong username or password") from e

    # create new token for the user
    access_token = orchestrator.create_access_token(
        data={"sub": user.username}, expires_min=120
    )
    return TokenModel(
        access_token=access_token, token_type="bearer", status=user.status
    )


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


@app.get("/version")
async def get_version() -> str:
    """
    Get the version of the server
    """
    return __version__


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
        "version": __version__,
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
