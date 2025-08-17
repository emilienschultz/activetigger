from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import (
    get_project,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import ComputeBertopicModel, UserInDBModel
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter()


@router.post("/bertopic/compute", dependencies=[Depends(verified_user)])
async def compute_bertopic(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bertopic: ComputeBertopicModel,
) -> None:
    """
    Compute BERTopic model for the project.
    """
    # Force the train dataset
    path_data = project.params.dir.joinpath(config.train_file)
    # Force the language of the project
    bertopic.language = project.params.language
    try:
        project.bertopic.compute(
            path_data=path_data,
            col_id=None,
            col_text="text",
            parameters=bertopic,
            name=bertopic.name,
            user=current_user.username,
        )
        orchestrator.log_action(current_user.username, "COMPUTE BERTopic MODEL", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/topics", dependencies=[Depends(verified_user)])
async def get_bertopic_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> list:
    """
    Get topics from the BERTopic model for the project.
    """
    try:
        return project.bertopic.get_topics(name=name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/projection", dependencies=[Depends(verified_user)])
async def get_bertopic_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list:
    """
    Get projection from the BERTopic model for the project.
    """
    try:
        projection = project.bertopic.get_projection()
        return projection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bertopic/delete", dependencies=[Depends(verified_user)])
async def delete_bertopic_model(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> None:
    """
    Delete a BERTopic model for the project.
    """
    try:
        project.bertopic.delete(name=name)
        orchestrator.log_action(current_user.username, "DELETE BERTopic MODEL", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
