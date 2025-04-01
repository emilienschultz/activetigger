import logging
from datetime import datetime
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
    UserGenerationComputing,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project
from activetigger.tasks.generate_call import GenerateCall

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/generate/models/available", dependencies=[Depends(verified_user)])
async def list_generation_models() -> list[GenerationModelApi]:
    """
    Returns the list of the available GenAI models for generation
    API (not the models themselves)
    """
    return orchestrator.db_manager.generations_service.get_available_models()


@router.get("/generate/models", dependencies=[Depends(verified_user)])
async def list_project_generation_models(
    project: Annotated[Project, Depends(get_project)],
) -> list[GenerationModel]:
    """
    Returns the list of the available GenAI models configure for a project
    """
    r = orchestrator.db_manager.generations_service.get_project_gen_models(project.name)
    return [GenerationModel(**i.__dict__) for i in r]


@router.post("/generate/models", dependencies=[Depends(verified_user)])
async def add_project_generation_models(
    project: Annotated[Project, Depends(get_project)], model: GenerationCreationModel
) -> int:
    """
    Add a new GenAI model for the project
    """

    try:
        # test if the model exists with this name for the project
        models = orchestrator.db_manager.generations_service.get_project_gen_models(
            project.name
        )
        for m in models:
            if m.name == model.name:
                raise HTTPException(
                    status_code=400, detail="A model with this name already exists"
                )

        # add the model
        return orchestrator.db_manager.generations_service.add_project_gen_model(
            project.name, model
        )
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
    return orchestrator.db_manager.generations_service.delete_project_gen_model(
        project.name, model_id
    )


@router.post("/generate/start", dependencies=[Depends(verified_user)])
async def postgenerate(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    request: GenerationRequest,
) -> None:
    """
    Launch a call to generate from a prompt
    Only one possible by user

    TODO : move to a module
    """

    try:
        # get subset of elements
        extract = project.schemes.get_sample(
            request.scheme, request.n_batch, request.mode
        )

        # get model
        model = orchestrator.db_manager.generations_service.get_gen_model(
            request.model_id
        )

        # add task to the queue
        unique_id = orchestrator.queue.add_task(
            "generation",
            project.name,
            GenerateCall(
                path_process=project.params.dir,
                username=current_user.username,
                project_slug=project.name,
                df=extract,
                prompt=request.prompt,
                model=GenerationModel(**model.__dict__),
            ),
        )

        project.computing.append(
            UserGenerationComputing(
                unique_id=unique_id,
                user=current_user.username,
                project=project.name,
                model_id=request.model_id,
                number=request.n_batch,
                time=datetime.now(),
                kind="generation",
                get_progress=GenerateCall.get_progress_callback(
                    project.params.dir.joinpath(unique_id)
                    if project.params.dir is not None
                    else None
                ),
            )
        )

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
        p = project.get_process("generation", current_user.username)
        if len(p) == 0:
            raise HTTPException(
                status_code=400, detail="No process found for this user"
            )
        unique_id = p[0].unique_id
        orchestrator.queue.kill(unique_id)
        orchestrator.log_action(
            current_user.username, "STOP GENERATE", project.params.project_slug
        )
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
    Get elements from prediction
    """
    try:
        # get data
        table = project.generations.get_generated(
            project.name, current_user.username, params.n_elements
        )

        # apply filters
        table["answer"] = project.generations.filter(table["answer"], params.filters)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail="Error in loading generated data" + str(e)
        )

    # join with the text
    # table = table.join(project.content["text"], on="index")

    r = table.to_dict(orient="records")
    return TableOutModel(items=r, total=len(r))


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
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
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
        # if no name, use the beginning of the text
        if prompt.name is not None:
            name = prompt.name
        else:
            name = prompt.text[0 : max(30, len(prompt.text))]

        # check if the name is already used
        if project.generations.prompt_exists(project.name, name):
            raise HTTPException(
                status_code=400, detail="A prompt with this name already exists"
            )

        # save prompt
        project.generations.save_prompt(
            current_user.username, project.name, prompt.text, name
        )
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate/prompts/delete", dependencies=[Depends(verified_user)])
async def delete_prompt(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    prompt_id: str,
) -> None:
    """
    Delete a prompt from the project
    """
    try:
        print(prompt_id, type(prompt_id))
        project.generations.delete_prompt(int(prompt_id))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
