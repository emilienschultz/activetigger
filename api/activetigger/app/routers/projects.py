from typing import Annotated, Any

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    check_auth_exists,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    AvailableProjectsModel,
    DatasetModel,
    ProjectAuthsModel,
    ProjectBaseModel,
    ProjectDescriptionModel,
    ProjectStateModel,
    ProjectUpdateModel,
    TestSetDataModel,
    UserInDBModel,
)
from activetigger.functions import slugify
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["projects"])


@router.post("/projects/close/{project_slug}", dependencies=[Depends(verified_user)])
async def close_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_slug: str,
) -> None:
    """
    Close a project from memory
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    try:
        orchestrator.stop_project(project_slug)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/{project_slug}/statistics", dependencies=[Depends(verified_user)])
async def get_project_statistics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str | None = None,
) -> ProjectDescriptionModel:
    """
    Statistics for a scheme and a user
    """
    try:
        return project.get_statistics(scheme=scheme, user=current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/auth", dependencies=[Depends(verified_user)])
async def get_project_auth(project_slug: str) -> ProjectAuthsModel:
    """
    Users auth on a project
    """
    if not orchestrator.exists(project_slug):
        raise HTTPException(status_code=404, detail="Project doesn't exist")
    try:
        return ProjectAuthsModel(auth=orchestrator.users.get_project_auth(project_slug))
    except Exception as e:
        raise HTTPException(status_code=500) from e


@router.post("/projects/new", dependencies=[Depends(verified_user)])
async def new_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: ProjectBaseModel,
) -> str:
    """
    Start the creation of a new project
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)
    # check if the project already exists
    try:
        project_slug = orchestrator.starting_project_creation(
            project=project,
            username=current_user.username,
        )
        orchestrator.log_action(current_user.username, "START CREATING PROJECT", project_slug)
        return project_slug
    except Exception as e:
        orchestrator.clean_unfinished_project(project_name=project.project_name)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/projects/update",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
async def update_project(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    update: ProjectUpdateModel,
) -> None:
    """
    Update a project
    - change the name
    - change the language
    - change context cols
    - change text cols
    - expand the number of elements in the trainset
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.update_project(update)
        orchestrator.log_action(current_user.username, "INFO UPDATE PROJECT", project.name)
        del orchestrator.projects[project.name]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
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
    test_rights(ServerAction.DELETE_PROJECT, current_user.username, project_slug)
    try:
        orchestrator.delete_project(project_slug)
    except Exception as e:
        print(f"Error deleting project {project_slug}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/projects/status", dependencies=[Depends(verified_user)])
async def get_project_status(
    project_name: str,
) -> str:
    """
    Get the status of a project
    - not existing
    - creating
    - existing
    """
    try:
        # if project is in creation
        if slugify(project_name) in orchestrator.project_creation_ongoing:
            return "creating"
        elif orchestrator.exists(project_name):
            return "existing"
        else:
            return "not existing"
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/projects/testset/{action}", dependencies=[Depends(verified_user)])
async def add_testdata(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    action: str,
    testset: TestSetDataModel | None = None,
) -> None:
    """
    Add a dataset for test when there is none available
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        if action == "create":
            if testset is None:
                raise Exception("No testset sent")
            project.add_testset(testset, current_user.username, project.name)
            orchestrator.log_action(current_user.username, "ADD TESTSET", project.name)
            return None
        if action == "delete":
            project.drop_testset()
            orchestrator.log_action(current_user.username, "DELETE TESTSET", project.name)
            return None
        raise Exception("action not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/projects")
async def get_projects(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> AvailableProjectsModel:
    """
    Get general informations on the server
    depending of the status of connected user
    """
    try:
        return AvailableProjectsModel(
            projects=orchestrator.get_auth_projects(current_user.username)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/datasets", dependencies=[Depends(verified_user)])
async def get_project_datasets(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list[DatasetModel]:
    """
    Get all datasets already available for a specific user
    """
    try:
        return orchestrator.get_auth_datasets(current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get(
    "/projects/{project_slug}",
    dependencies=[Depends(verified_user), Depends(check_auth_exists)],
)
async def get_project_state(
    project: Annotated[Project, Depends(get_project)],
) -> ProjectStateModel:
    """
    Get the state of a specific project
    """
    try:
        return project.state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
