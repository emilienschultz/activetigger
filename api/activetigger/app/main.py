import importlib
import logging
from contextlib import asynccontextmanager
from importlib.abc import Traversable
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
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
    files,
    generation,
    models,
    projects,
    schemes,
    users,
)
from activetigger.datamodels import (
    ServerStateModel,
    TableOutModel,
    TokenModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator

# to log specific events from api
logger = logging.getLogger("api")
logger_simplemodel = logging.getLogger("simplemodel")


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

# add static folder
app.mount("/static", StaticFiles(directory=orchestrator.path.joinpath("static")), name="static")

# add routers
app.include_router(users.router)
app.include_router(projects.router)
app.include_router(annotations.router)
app.include_router(schemes.router)
app.include_router(features.router)
app.include_router(export.router)
app.include_router(models.router)
app.include_router(generation.router)
app.include_router(files.router)


# allow multiple servers (avoir CORS error)
app.add_middleware(
    CORSMiddleware,
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
    Welcome page for the API
    """
    data_path: Traversable = importlib.resources.files("activetigger")
    with open(str(data_path.joinpath("html", "welcome.html")), "r") as f:
        r = f.read()
    return r


@app.get("/version")
async def get_version() -> str:
    """
    Get the version of the server
    """
    return __version__


@app.get("/server")
async def get_queue() -> ServerStateModel:
    """
    Get the state of the server
    """
    return orchestrator.server_state


@app.post("/token")
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
) -> TokenModel:
    """
    Authentificate user from username/password and return token
    """

    try:
        # authentificate the user
        user = orchestrator.users.authenticate_user(form_data.username, form_data.password)
        # create new token for the user
        access_token = orchestrator.create_access_token(
            data={"sub": user.username}, expires_min=120
        )
        return TokenModel(access_token=access_token, token_type="bearer", status=user.status)
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


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


@app.post("/kill", dependencies=[Depends(verified_user)])
async def kill_process(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    unique_id: str,
) -> None:
    """
    Kill a process with a unique id
    """
    test_rights("kill process", current_user.username)
    try:
        orchestrator.queue.kill(unique_id)
        orchestrator.log_action(current_user.username, f"KILL PROCESS: {unique_id}", "all")
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
