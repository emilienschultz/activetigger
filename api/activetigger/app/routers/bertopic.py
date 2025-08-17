from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from activetigger.app.dependencies import (
    get_project,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import ComputeBertTopicModel, UserInDBModel
from activetigger.project import Project

router = APIRouter()


@router.post("/bertopic/compute", dependencies=[Depends(verified_user)])
async def compute_bertopic(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bertopic: ComputeBertTopicModel,
) -> None:
    """
    Compute BERTopic model for the project.
    """
    # For the moment, only on the train file
    path_data = project.params.dir.joinpath(config.train_file)
    print(path_data)
    try:
        project.bertopic.compute(
            path_data=path_data,
            col_id=None,
            col_text="text",
            parameters=bertopic,
            name=bertopic.name,
            user=current_user.username,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/topics", dependencies=[Depends(verified_user)])
async def get_bertopic_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> list:
    """
    Get topics from the BERTopic model for the project.
    """
    try:
        topics = project.bertopic.get_topics()
        return topics
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
