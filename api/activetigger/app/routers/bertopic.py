from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import (
    get_project,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import BertopicTopicsOutModel, ComputeBertopicModel, UserInDBModel
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter()


@router.post("/bertopic/compute", dependencies=[Depends(verified_user)])
async def compute_bertopic(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bertopic: ComputeBertopicModel,
) -> str:
    """
    Compute BERTopic model for the project.
    """
    # Force the train dataset
    path_data = project.params.dir.joinpath(config.train_file)  # type: ignore
    # Force the language of the project
    bertopic.language = project.params.language
    if not project.bertopic.name_available(bertopic.name):
        raise HTTPException(
            status_code=400,
            detail=f"BERTopic model with name '{bertopic.name}' already exists (after slugification).",
        )
    try:
        unique_id = project.bertopic.compute(
            path_data=path_data,
            col_id=None,
            col_text="text",
            parameters=bertopic,
            name=bertopic.name,
            user=current_user.username,
            force_compute_embeddings=bertopic.force_compute_embeddings,
        )
        orchestrator.log_action(
            current_user.username, f"COMPUTE BERTopic MODEL: {bertopic.name}", project.name
        )
        return unique_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/topics", dependencies=[Depends(verified_user)])
async def get_bertopic_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> BertopicTopicsOutModel:
    """
    Get topics from the BERTopic model for the project.
    """
    try:
        return BertopicTopicsOutModel(
            topics=project.bertopic.get_topics(name=name),
            parameters=project.bertopic.get_parameters(name=name),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/bertopic/projection", dependencies=[Depends(verified_user)])
async def get_bertopic_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> dict[str, list | dict]:
    """
    Get projection from the BERTopic model for the project.
    """
    try:
        return project.bertopic.get_projection(name=name)
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
        orchestrator.log_action(
            current_user.username, f"DELETE BERTopic MODEL: {name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
