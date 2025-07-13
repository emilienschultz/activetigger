import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    get_project,
    verified_user,
)
from activetigger.datamodels import (
    GeneratedElementsIn,
    GenerationCreationModel,
    GenerationModel,
    GenerationModelApi,
    GenerationRequest,
    PromptInputModel,
    PromptModel,
    TableOutModel,
    UserInDBModel,
)
from activetigger.generation.generations import Generations
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/generate/models/available", dependencies=[Depends(verified_user)])
async def list_generation_models() -> list[GenerationModelApi]:
    """
    Returns the list of the available GenAI models for generation
    API (not the models themselves)
    """
    try:
        return Generations.get_available_models()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/models", dependencies=[Depends(verified_user)])
async def list_project_generation_models(
    project: Annotated[Project, Depends(get_project)],
) -> list[GenerationModel]:
    """
    Returns the list of the available GenAI models configure for a project
    """
    try:
        return project.generations.available_models(project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/models", dependencies=[Depends(verified_user)])
async def add_project_generation_models(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model: GenerationCreationModel,
) -> int:
    """
    Add a new GenAI model for the project
    """
    try:
        # test if the model exists with this name for the project
        if project.generations.model_exists(project.name, model.name):
            raise HTTPException(status_code=400, detail="A model with this name already exists")

        # add the model
        r = project.generations.add_model(project.name, model, current_user.username)
        return r
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/generate/models/{model_id}",
    dependencies=[Depends(verified_user)],
)
async def delete_project_generation_models(
    project: Annotated[Project, Depends(get_project)], model_id: int
) -> None:
    """
    Delete a GenAI model from the project
    """
    try:
        project.generations.delete_model(project.name, model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/start", dependencies=[Depends(verified_user)])
async def postgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenerationRequest,
) -> None:
    """
    Launch a call to generate from a prompt
    """

    try:
        project.start_generation(request, current_user.username)
        orchestrator.log_action(
            current_user.username,
            "START GENERATE",
            project.params.project_slug,
        )
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/stop", dependencies=[Depends(verified_user)])
async def stop_generation(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop current generation
    """
    try:
        orchestrator.stop_process(
            current_user.username,
            kind=["generation"],
        )
        orchestrator.log_action(current_user.username, "STOP GENERATE", project.params.project_slug)
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/elements", dependencies=[Depends(verified_user)])
async def getgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    params: GeneratedElementsIn,
) -> TableOutModel:
    """
    Get elements generated
    """
    try:
        # get data
        table = project.generations.get_generated(
            project.name, current_user.username, params.n_elements
        )

        # apply filters
        table["answer"] = project.generations.filter(table["answer"], params.filters)

        # join with the text
        # table = table.join(project.content["text"], on="index")

        return TableOutModel(items=table.to_dict(orient="records"), total=len(table))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error in loading generated data" + str(e))


@router.post("/generate/elements/drop", dependencies=[Depends(verified_user)])
async def dropgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Drop all elements from prediction for a user
    """
    try:
        project.generations.drop_generated(project.name, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/generate/prompts", dependencies=[Depends(verified_user)])
async def get_prompts(
    project: Annotated[Project, Depends(get_project)],
) -> list[PromptModel]:
    """
    Get the list of prompts for the user
    """
    try:
        return project.generations.get_prompts(project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/prompts/add", dependencies=[Depends(verified_user)])
async def add_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt: PromptInputModel,
) -> None:
    """
    Add a prompt to the project
    """
    try:
        project.generations.save_prompt(prompt, current_user.username, project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/prompts/delete", dependencies=[Depends(verified_user)])
async def delete_prompt(
    project: Annotated[Project, Depends(get_project)],
    prompt_id: str,
) -> None:
    """
    Delete a prompt from the project
    """
    try:
        project.generations.delete_prompt(int(prompt_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
