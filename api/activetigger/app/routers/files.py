import asyncio
import os
import random
import shutil
from pathlib import Path
from typing import Annotated

import aiofiles  # type: ignore[import]
from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from activetigger.app.dependencies import (
    ProjectAction,
    ServerAction,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["files"])


@router.post("/files/add/project")
async def upload_file_project(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server to create a new project
    use: type de file
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)

    # add a delay if projects are already being created
    if len(orchestrator.project_creation_ongoing) >= 3:
        await asyncio.sleep(random.randint(1, 4))

    # check if the project does not already exist
    if orchestrator.exists(project_name):
        raise HTTPException(
            status_code=500, detail="Project already exists, please choose another name"
        )
    # test the incoming file
    if file.filename is None:
        raise HTTPException(status_code=500, detail="Problem with the file")
    if (
        not file.filename.endswith("csv")
        and not file.filename.endswith("parquet")
        and not file.filename.endswith("xlsx")
    ):
        raise HTTPException(status_code=500, detail="Only csv and parquet files are allowed")

    # try to upload the file
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        project_path = Path(f"{config.data_path}/projects/{project_slug}")
        os.makedirs(project_path)

        # Read and write the file asynchronously
        async with aiofiles.open(project_path.joinpath(file.filename), "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
        print("File uploaded successfully")

    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():
            project_path.rmdir()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/add/dataset")
async def upload_file_dataset(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server for a project in the data folder
    """
    test_rights(ProjectAction.MANAGE_FILES, current_user.username)
    # test the incoming file
    if file.filename is None:
        raise HTTPException(status_code=500, detail="Problem with the file")
    if (
        not file.filename.endswith("csv")
        and not file.filename.endswith("parquet")
        and not file.filename.endswith("xlsx")
    ):
        raise HTTPException(status_code=500, detail="Only csv and parquet files are allowed")

    try:
        async with aiofiles.open(
            project.data.path_datasets.joinpath(file.filename), "wb"
        ) as out_file:
            while chunk := await file.read(1024 * 1024):
                await out_file.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/copy/project")
async def copy_existing_data(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    source_project: str,
    from_toy: bool
) -> None:
    """
    Copy an existing project to create a new one

    if copy dataset from toy datasets: projects/toy-datasets/NAME.parquet
    if copy from project: projects/NAME/data_all.parquet
    """
    test_rights(ServerAction.CREATE_PROJECT, current_user.username)

    # check if the project does not already exist
    if orchestrator.exists(project_name):
        raise HTTPException(
            status_code=500, detail="Project already exists, please choose another name"
        )
    if from_toy:
        if not orchestrator.exists(source_project,is_toy_dataset=True):
            raise HTTPException(status_code=500, detail="Source project does not exist")
        source_path = Path(f"{config.data_path}/projects/toy-datasets/{source_project}.parquet")
    else:
        # check if the source project exists
        if not orchestrator.exists(source_project):
            raise HTTPException(status_code=500, detail="Source project does not exist")
        source_path = Path(f"{config.data_path}/projects/{source_project}/{config.data_all}")
    
    # try to copy the project
    try:
        # create a folder for the project to be created
        project_slug = orchestrator.check_project_name(project_name)
        
        project_path = Path(f"{config.data_path}/projects/{project_slug}")
        os.makedirs(project_path)

        # copy the full dataset
        shutil.copyfile(
            source_path,
            project_path.joinpath(config.data_all),
        )

    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():
            shutil.rmtree(project_path)
        raise HTTPException(status_code=500, detail=str(e))