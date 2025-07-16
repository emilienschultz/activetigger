from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import (
    ProjectAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    FeatureDescriptionModel,
    FeatureModel,
    UserInDBModel,
    WaitingModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter()


@router.get("/features", dependencies=[Depends(verified_user)])
async def get_features(project: Annotated[Project, Depends(get_project)]) -> list[str]:
    """
    Available features for the project
    """
    return list(project.features.map.keys())


@router.post("/features/add", dependencies=[Depends(verified_user)])
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
    test_rights(ProjectAction.MODIFY_PROJECT, current_user.username, project.name)
    df = project.content["text"]
    try:
        project.features.compute(
            df, feature.name, feature.type, feature.parameters, current_user.username
        )
        orchestrator.log_action(
            current_user.username, f"COMPUTE FEATURE: {feature.type}", project.name
        )
        return WaitingModel(detail=f"computing {feature.type}, it could take a few minutes")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/features/delete", dependencies=[Depends(verified_user)])
async def delete_feature(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> None:
    """
    Delete a specific feature
    """
    test_rights(ProjectAction.MODIFY_PROJECT, current_user.username, project.name)
    try:
        project.features.delete(name)
        orchestrator.log_action(current_user.username, f"DELETE FEATURE: {name}", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/available", dependencies=[Depends(verified_user)])
async def get_feature_info(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict[str, FeatureDescriptionModel]:
    """
    Get feature info
    """
    try:
        return project.features.get_available()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
