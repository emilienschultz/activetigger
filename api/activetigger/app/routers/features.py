from typing import Annotated

import pandas as pd
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
    FeatureDescriptionModelOut,
    FeatureModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["features"])


@router.post("/features/add", dependencies=[Depends(verified_user)])
async def post_embeddings(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    feature: FeatureModel,
):
    """
    Compute features :
    - same prcess
    - specific process : function + temporary file + update
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        # gather all text data to compute features on
        if project.data.train is None:
            raise HTTPException(
                status_code=400, detail="No training data available to compute features."
            )
        series_list = [project.data.train["text"]]
        if project.data.valid is not None:
            series_list.append(project.data.valid["text"])
        if project.data.test is not None:
            series_list.append(project.data.test["text"])
        df = pd.concat(series_list)

        # compute features
        project.features.compute(
            df, feature.name, feature.type, feature.parameters, current_user.username
        )
        orchestrator.log_action(
            current_user.username, f"COMPUTE FEATURE: {feature.type}", project.name
        )
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
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        project.features.delete(name)
        orchestrator.log_action(current_user.username, f"DELETE FEATURE: {name}", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/available", dependencies=[Depends(verified_user)])
async def get_feature_info(
    project: Annotated[Project, Depends(get_project)],
) -> dict[str, FeatureDescriptionModelOut]:
    """
    Get feature info
    """
    try:
        return project.features.get_available()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
