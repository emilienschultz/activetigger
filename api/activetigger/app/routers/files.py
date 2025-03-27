import logging
import os
from pathlib import Path
from typing import Annotated, List

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    UploadFile,
)
from slugify import slugify

from activetigger.app.dependencies import (
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    UserInDBModel,
)

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
        files = os.listdir(f"{os.environ['ACTIVETIGGER_PATH']}/upload")
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
    if username is None:
        raise Exception("No username provided")
    if file.filename is None:
        raise Exception("No filename provided")
    print("start writing the project")

    # create a folder for the project to be created
    project_slug = slugify(project_name)
    project_path = Path(f"{os.environ['ACTIVETIGGER_PATH']}/{project_slug}")
    if project_path.exists():
        raise Exception("Project already exists")
    os.makedirs(project_path)

    # Read and write the file asynchronously
    try:
        with (project_path.joinpath(file.filename)).open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # Read in 1MB chunks
                print("writing chunk")
                buffer.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File writing error: {str(e)}")

    print("end writing the project")


@router.post("/files/add/project")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project_name: str,
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server
    use: type de file
    """
    test_rights("manage files", current_user.username)

    # test the incoming file
    if file.filename is None:
        raise HTTPException(status_code=500, detail="Problem with the file")
    if (
        not file.filename.endswith("csv")
        and not file.filename.endswith("parquet")
        and not file.filename.endswith("xlsx")
    ):
        raise HTTPException(
            status_code=500, detail="Only csv and parquet files are allowed"
        )
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
        file_path = Path(f"{os.environ['ACTIVETIGGER_PATH']}/upload/{filename}")
        if file_path.exists():
            file_path.unlink()
            return None
        else:
            raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
