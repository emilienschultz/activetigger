import time
from enum import Enum
from typing import Annotated

from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError  # type: ignore[import]

from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project


def get_orchestrator():
    return orchestrator


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


async def get_project(project_slug: str) -> Project:
    """
    Dependency to get existing project
    - if already loaded, return it
    - if not loaded, load it first
    """

    # test if project exists
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project not found")
    try:
        if project_slug not in orchestrator.projects:
            orchestrator.manage_fifo_queue()
            orchestrator.start_project(project_slug)
        return orchestrator.projects[project_slug]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


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
        return orchestrator.users.get_user(name=username)
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
    try:
        auth = orchestrator.users.auth(current_user.username, project_slug)
        if not auth:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid rights")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid rights") from e


# --------------------------
# Rights management
# --------------------------


class ServerAction(str, Enum):
    MANAGE_USERS = "manage users"
    MANAGE_SERVER = "manage server"
    KILL_PROCESS = "kill process"
    CREATE_PROJECT = "create project"
    DELETE_PROJECT = "delete project"


class ProjectAction(str, Enum):
    ADD = "add to project"
    DELETE = "delete from project"
    UPDATE = "update project"
    GET = "get project information"
    ADD_ANNOTATION = "add annotation"
    UPDATE_ANNOTATION = "modify annotation"
    EXPORT_DATA = "export data"
    MANAGE_FILES = "manage files"
    MONITOR = "access specific project information"
    GENERATE = "use generation features"


def test_rights(
    action: ServerAction | ProjectAction,
    username: str,
    project_slug: str | None = None,
    scheme: str | None = None,
) -> bool:
    """
    Management of rights on the routes

    Based on an action, a user, a project
    Existing status : root, manager, annotator, demo
    Existing rights : manager, contributor
    Not implemented : rights specific to scheme
    """
    try:
        user = orchestrator.users.get_user(name=username)
    except Exception as e:
        raise HTTPException(404) from e

    if action not in ServerAction and action not in ProjectAction:
        raise HTTPException(
            status_code=500,
            detail=f"Action {action} is not a valid action",
        )

    # general status
    status = user.status

    # root user can do anything
    if status == "root":
        return True

    match action:
        case ServerAction.CREATE_PROJECT | ServerAction.DELETE_PROJECT:
            if status in ["manager"]:
                return True

    # specific case demo
    if status == "demo":
        if action in [ProjectAction.GET, ProjectAction.ADD_ANNOTATION]:
            return True
        else:
            raise HTTPException(
                status_code=408,
                detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
            )

    # Get auth for the project
    if not project_slug:
        raise HTTPException(500, "Project name missing")
    auth = orchestrator.users.auth(username, project_slug)
    if auth is None:
        raise HTTPException(
            status_code=408,
            detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
        )

    match action:
        # only manager can delete/modify elements
        case (
            ProjectAction.DELETE
            | ProjectAction.UPDATE
            | ProjectAction.UPDATE_ANNOTATION
            | ProjectAction.EXPORT_DATA
            | ProjectAction.MANAGE_FILES
        ):
            if auth in ["manager"]:
                return True
        # only manager and contributor can create
        case ProjectAction.ADD | ProjectAction.MONITOR | ProjectAction.GENERATE:
            if auth in ["manager", "contributor"]:
                return True
        # everyone can get info or add annotation
        case ProjectAction.ADD_ANNOTATION | ProjectAction.GET:
            return True

    # by default, no rights
    raise HTTPException(
        status_code=408,
        detail=f"Forbidden: User {username} has no rights to perform action {action} on project {project_slug}",
    )
