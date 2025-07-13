#####
# Those route are not used for the moment
#####
import asyncio
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Annotated, List

import aiofiles
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)

from activetigger.app.dependencies import (
    test_rights,
    verified_user,
)
from activetigger.config import config
from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


@router.get("/files")
async def get_files(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> List[str]:
    """
    Get all files
    """
    try:
        files = os.listdir(f"{config.data_path}/projects/upload")
        if current_user.status == "root":
            return files
        else:
            return [i for i in files if i.startswith(current_user.username)]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def save_uploaded_file(file: UploadFile, file_path: Path, project_path: Path):
    try:
        async with aiofiles.open(file_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                await out_file.write(chunk)
    except Exception as e:
        # Cleanup on failure
        if project_path.exists():
            shutil.rmtree(project_path)
        print(f"Failed to write file: {e}")


@router.post("/files/add/project")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server to create a new project
    use: type de file
    """
    test_rights("manage files", current_user.username)

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

        # setting the project in creation
        orchestrator.starting_project_creation(project_slug)
        os.makedirs(project_path)
        file_path = project_path.joinpath(file.filename)

        background_tasks.add_task(save_uploaded_file, file, file_path, project_path)

    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():
            project_path.rmdir()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/files/delete")
async def delete_file(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    filename: str,
) -> None:
    """
    Delete a file
    """
    test_rights("manage files", current_user.username)
    try:
        file_path = Path(f"{config.data_path}/projects/upload/{filename}")
        if file_path.exists():
            file_path.unlink()
            return None
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
