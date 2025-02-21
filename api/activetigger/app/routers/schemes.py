from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    ActionModel,
    CodebookModel,
    SchemeModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter()


@router.post("/schemes/label/rename", dependencies=[Depends(verified_user)])
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

    print("old label deleted")

    # log
    orchestrator.log_action(
        current_user.username,
        f"RENAME LABEL in {scheme}: label {former_label} to {new_label}",
        project.name,
    )
    return None


@router.post("/schemes/label/{action}", dependencies=[Depends(verified_user)])
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


@router.post("/schemes/codebook", dependencies=[Depends(verified_user)])
async def post_codebook(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    codebook: CodebookModel,
) -> None:
    """
    Add codebook
    """
    test_rights("modify project element", current_user.username, project.name)

    try:
        project.schemes.add_codebook(codebook.scheme, codebook.content, codebook.time)
        orchestrator.log_action(
            current_user.username,
            f"MODIFY CODEBOOK: codebook {codebook.scheme}",
            project.name,
        )
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/schemes/codebook", dependencies=[Depends(verified_user)])
async def get_codebook(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
) -> CodebookModel:
    """
    Get the codebook of a scheme for a project
    """
    try:
        r = project.schemes.get_codebook(scheme)
        return CodebookModel(
            scheme=scheme,
            content=str(r["codebook"]),
            time=str(r["time"]),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/schemes/{action}", dependencies=[Depends(verified_user)])
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
