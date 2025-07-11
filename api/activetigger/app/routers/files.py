#####
# Those route are not used for the moment
#####

import logging
import os
import random
import time
from pathlib import Path
from typing import Annotated, List

from fastapi import (
    APIRouter,
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


async def new_project_file(file: UploadFile, username: str, project_name: str) -> None:
    """
    Stream the uploaded file in a new project folder with the same name
    """
    # create a folder for the project to be created
    project_slug = orchestrator.check_project_name(project_name)
    project_path = Path(f"{config.data_path}/projects/{project_slug}")

    # setting the project in creation
    orchestrator.starting_project_creation(project_slug)

    if file.filename is None:
        raise Exception("Problem with the file name")

    try:
        os.makedirs(project_path)

        # Read and write the file asynchronously
        with (project_path.joinpath(file.filename)).open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                print("writing chunk")
                buffer.write(chunk)
    except Exception as e:
        # if failed, remove the project folder
        if project_path.exists():
            project_path.rmdir()
        raise HTTPException(status_code=500, detail=f"File writing error: {str(e)}")


@router.post("/files/add/project")
async def upload_file(
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
        time.sleep(random.randint(1, 4))

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
        await new_project_file(file, current_user.username, project_name)
        return None
    except Exception as e:
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
