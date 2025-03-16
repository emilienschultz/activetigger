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


async def save_file(file: UploadFile, user: str):
    """Stream the uploaded file and save it to disk."""
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    if not os.path.exists(f"{os.environ['ACTIVETIGGER_PATH']}/upload"):
        os.makedirs(f"{os.environ['ACTIVETIGGER_PATH']}/upload")
    filename = str(user) + "_" + str(int(time.time())) + "_" + file.filename
    file_path = Path(f"{os.environ['ACTIVETIGGER_PATH']}/upload/{filename}")
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File saved: {file_path}")


@router.post("/files/add")
async def upload_file(
    background_tasks: BackgroundTasks,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    file: UploadFile = File(...),
) -> None:
    """Accepts large files via streaming and saves asynchronously."""
    test_rights("delete file", current_user.username)
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
    test_rights("delete file", current_user.username)
    try:
        file_path = Path(f"{os.environ['ACTIVETIGGER_PATH']}/upload/{filename}")
        if file_path.exists():
            file_path.unlink()
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/files")
async def get_files() -> List[str]:
    """
    Get all files
    """
    try:
        files = os.listdir(f"{os.environ['ACTIVETIGGER_PATH']}/upload")
        return files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
