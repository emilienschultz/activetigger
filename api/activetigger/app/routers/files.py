import logging
import os
import shutil
import time
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


async def save_file(file: UploadFile, username: str):
    """
    Stream the uploaded file and save it to disk.
    For the moment
    /upload folder
    /user folder by user
    """
    if username is None:
        raise HTTPException(status_code=401, detail="User not found")
    folder_path = Path(f"{os.environ['ACTIVETIGGER_PATH']}/upload")
    if not folder_path.exists():
        os.makedirs(folder_path)
    filename = f"{username}_{int(time.time())}_{file.filename}"
    with folder_path.joinpath(filename).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)


@router.post("/files/add")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: UploadFile = File(...),
) -> None:
    """
    Upload a file on the server
    """
    test_rights("manage files", current_user.username)
    background_tasks.add_task(save_file, file, "dev")
    return None


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
