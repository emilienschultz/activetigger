import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from activetigger.app.dependencies import ProjectAction, get_project, test_rights, verified_user
from activetigger.datamodels import (
    BertopicTopicsOutModel,
    ComputeBertopicModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["BERTopic"])


@router.post("/bertopic/compute", dependencies=[Depends(verified_user)])
async def compute_bertopic(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bertopic: ComputeBertopicModel,
) -> str:
    """
    Compute BERTopic model for the project.
    """

    def _impl():
        if not project.bertopic.name_available(bertopic.name):
            raise HTTPException(
                status_code=400,
                detail=f"BERTopic model with name '{bertopic.name}' already exists (after slugification).",
            )
        if project.params.dir is None:
            raise HTTPException(
                status_code=400,
                detail="Project dataset path is not set. Cannot compute BERTopic model.",
            )

        # Force the language of the project
        bertopic.language = project.params.language

        try:
            unique_id = project.bertopic.compute(
                path_data=project.params.dir,
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

    return await asyncio.to_thread(_impl)


@router.get("/bertopic/topics", dependencies=[Depends(verified_user)])
async def get_bertopic_topics(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> BertopicTopicsOutModel:
    """
    Get topics from the BERTopic model for the project.
    """

    def _impl():
        try:
            return BertopicTopicsOutModel(
                topics=project.bertopic.get_topics(name=name),
                parameters=project.bertopic.get_parameters(name=name),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.get("/bertopic/projection", dependencies=[Depends(verified_user)])
async def get_bertopic_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> dict[str, list | dict]:
    """
    Get projection from the BERTopic model for the project.
    """

    def _impl():
        try:
            return project.bertopic.get_projection(name=name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.post("/bertopic/delete", dependencies=[Depends(verified_user)])
async def delete_bertopic_model(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(...),
) -> None:
    """
    Delete a BERTopic model for the project.
    """

    def _impl():
        try:
            project.bertopic.delete(name=name)
            orchestrator.log_action(
                current_user.username, f"DELETE BERTopic MODEL: {name}", project.name
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.post("/bertopic/export-to-scheme", dependencies=[Depends(verified_user)])
async def export_bertopic_to_scheme(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    topic_model_name: str = Query(...),
) -> None:
    """
    Export the topic model as a scheme for the train set
    """

    def _impl():
        try:
            test_rights(ProjectAction.ADD, current_user.username, project.name)

            labels, clusters, topic_id_to_topic_name = project.bertopic.export_to_scheme(
                topic_model_name
            )

            new_scheme_name = f"topic-model-{topic_model_name}"

            # add a new scheme
            project.schemes.add_scheme(
                name=new_scheme_name,
                labels=labels,
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

    return await asyncio.to_thread(_impl)
