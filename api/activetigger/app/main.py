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
from activetigger.app.dependencies import ProjectAction, ServerAction, test_rights, verified_user
from activetigger.app.routers import (
    annotations,
    bertopic,
    export,
    features,
    files,
    generation,
    messages,
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
logger_quickmodel = logging.getLogger("quickmodel")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Frame the execution of the api
    """
    print("Active Tigger starting")
    yield
    print("Active Tigger closing")


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
app.include_router(bertopic.router)
app.include_router(messages.router)


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


@app.post("/server/restart", dependencies=[Depends(verified_user)])
async def restart_queue(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Restart the queue & the memory
    """
    test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    try:
        orchestrator.reset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
        user = orchestrator.users.authenticate_user(form_data.username, form_data.password)
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
        test_rights(ServerAction.MANAGE_SERVER, current_user.username)
    else:
        test_rights(ProjectAction.GET, current_user.username, project_slug)
    df = orchestrator.get_logs(project_slug, limit)
    return TableOutModel(
        items=df.to_dict(orient="records"),
        total=limit,
    )


@app.post("/stop", dependencies=[Depends(verified_user)])
async def stop_process(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    unique_id: str | None = None,
    kind: str | None = None,
) -> None:
    """
    Stop processes either by unique_id or by kind for a user
    - unique_id: stop a specific process (only for administrator)
    - kind: stop all processes of a given kind for the user
    """
    if unique_id is None and kind is None:
        raise HTTPException(status_code=400, detail="You must provide a unique_id or a kind")
    try:
        if unique_id is not None:
            test_rights(ServerAction.MANAGE_SERVER, current_user.username)
            orchestrator.stop_process(unique_id, current_user.username)
        if kind is not None:
            orchestrator.stop_user_processes(kind, current_user.username)
        orchestrator.log_action(
            current_user.username,
            f"STOP PROCESS: {kind if kind is not None else unique_id}",
            "general",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
