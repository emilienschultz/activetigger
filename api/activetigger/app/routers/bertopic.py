from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.config import config
from activetigger.datamodels import (
    BertopicTopicsOutModel,
    ComputeBertopicModel,
    TopicsOutModel,
    UserInDBModel,
)
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
        r = project.bertopic.get_projection(name=name)
        print(r)
        return r
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


@router.post("/bertopic/export-to-scheme", dependencies=[Depends(verified_user)])
async def export_bertopoc_to_scheme(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    topic_model_name: str = Query(...),
) -> None:
    """
    Export the topic model as a scheme for the train set
    """
    try:
        test_rights(ProjectAction.ADD_ANNOTATION, current_user.username, project.name)

        # Retrieve topics and clusters
        topics = project.bertopic.get_topics(topic_model_name)

        def get_topic_id(t: str) -> int:
            return int(t.split("_")[0])

        topic_id_to_topic_name = {
            get_topic_id(topic.Name): topic.Name
            for topic in topics
            if get_topic_id(topic.Name) != -1
        }
        clusters: dict[str, int] = project.bertopic.get_clusters(topic_model_name)

        new_scheme_name = f"topic-model-{topic_model_name}"
        project.schemes.add_scheme(
            name=new_scheme_name,
            labels=[topic.Name for topic in topics if get_topic_id(topic.Name) != -1],
            user=current_user.username,
        )
        # Transform the annotation into the right format
        elements = [
            {"element_id": el_id, "annotation": topic_id_to_topic_name[cluster], "comment": ""}
            for (el_id, cluster) in clusters.items()
            if cluster != -1
        ]
        project.schemes.projects_service.add_annotations(
            dataset="train",
            user_name=current_user.username,
            project_slug=project.name,
            scheme=new_scheme_name,
            elements=elements,
        )

        orchestrator.log_action(
            current_user.username, f"Export BERTopic to scheme : {new_scheme_name}", project.name
        )

    except Exception as e:
        orchestrator.log_action(current_user.username, f"DEBUG-EXPORT-TO-SCHEME: {e}", project.name)
        raise HTTPException(status_code=500, detail=str(e))
