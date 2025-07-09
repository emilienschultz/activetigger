import time
from typing import Annotated

from fastapi import (
    Depends,
    HTTPException,
    Request,
)
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError

from activetigger.datamodels import (
    ProjectModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator


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

    # server operation
    if action == "server operation":
        if status in ["root"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # possibility to create project
    if action == "create project":
        if status in ["root", "manager"]:
            return True
        else:
            raise HTTPException(500, "No rights for this action")

    # manage files
    if action == "manage files":
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
