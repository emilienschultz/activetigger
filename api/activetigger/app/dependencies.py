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


def manage_fifo_queue():
    """
    Manage the current projects in memory
    """
    if len(orchestrator.projects) >= orchestrator.max_projects:
        old_element = sorted(
            [[p, orchestrator.projects[p].starting_time] for p in orchestrator.projects],
            key=lambda x: x[1],
        )[0]
        if (
            old_element[1] < time.time() - 600
        ):  # check if the project has a least ten minutes old to avoid destroying current projects
            del orchestrator.projects[old_element[0]]
            print(f"Delete project {old_element[0]} to gain memory")
        else:
            print("Too many projects in the current memory")
            raise HTTPException(
                status_code=500,
                detail="There is too many projects currently loaded in this server. Please wait",
            )


async def get_project(project_slug: str) -> Project:
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

    # Manage a FIFO queue
    manage_fifo_queue()

    # load the project
    try:
        orchestrator.start_project(project_slug)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="There is a problem with the project loading. Please contact the administrator"
            + str(e),
        )

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
    try:
        auth = orchestrator.users.auth(current_user.username, project_slug)
        if not auth:
            raise HTTPException(status_code=403, detail="Forbidden: Invalid rights")
    except Exception as e:
        raise HTTPException(status_code=403, detail="Forbidden: Invalid rights") from e


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


def test_rights(
    action: ServerAction | ProjectAction,
    username: str,
    project_slug: str | None = None,
    scheme: str | None = None,
) -> bool:
    """
    Management of rights on the routes

    Based on an action, a user, a project
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
        case ProjectAction.ADD:
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


def check_storage(username: str) -> None:
    """
    Check if the user storage is not exceeded
    """
    limit = orchestrator.users.get_storage_limit(username)
    if orchestrator.users.get_storage(username) > limit * 1000:
        raise HTTPException(
            status_code=500,
            detail=f"User storage limit exceeded ({limit} Gb), please delete some models in your projects",
        )
