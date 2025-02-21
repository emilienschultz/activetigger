from io import StringIO
from typing import Annotated

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Response,
)
from fastapi.responses import FileResponse, StreamingResponse

from activetigger.app.dependencies import (
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    UserInDBModel,
)
from activetigger.project import Project

router = APIRouter(tags=["export"])


@router.get("/export/data", dependencies=[Depends(verified_user)])
async def export_data(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
    format: str,
    dataset: str = "train",
) -> FileResponse:
    """
    Export labelled data
    """
    r = project.export_data(format=format, scheme=scheme, dataset=dataset)
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


@router.get("/export/features", dependencies=[Depends(verified_user)])
async def export_features(
    project: Annotated[Project, Depends(get_project)],
    features: list = Query(),
    format: str = Query(),
) -> FileResponse:
    """
    Export features
    """
    try:
        r = project.export_features(features=features, format=format)
        return FileResponse(r["path"], filename=r["name"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/prediction/simplemodel", dependencies=[Depends(verified_user)])
async def export_simplemodel_predictions(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    format: str = "csv",
) -> StreamingResponse:
    """
    Export prediction simplemodel for the project/user/scheme if any
    """
    try:
        output, headers = project.simplemodels.export_prediction(
            scheme, current_user.username, format
        )
        return StreamingResponse(output, headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/prediction", dependencies=[Depends(verified_user)])
async def export_prediction(
    project: Annotated[Project, Depends(get_project)],
    format: str = Query(),
    name: str = Query(),
) -> FileResponse:
    """
    Export annotations
    """
    r = project.bertmodels.export_prediction(
        name=name, file_name="predict_all.parquet", format=format
    )
    if "error" in r:
        raise HTTPException(status_code=500, detail=r["error"])
    return FileResponse(r["path"], filename=r["name"])


@router.get("/export/bert", dependencies=[Depends(verified_user)])
async def export_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str = Query(),
) -> str:
    """
    Export fine-tuned BERT model
    """
    test_rights("modify project", current_user.username, project.name)
    try:
        r = project.bertmodels.export_bert(name=name)
        return "/static/" + r["name"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/raw", dependencies=[Depends(verified_user)])
async def export_raw(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> dict:
    """
    Export raw data of the project
    """
    test_rights("modify project", current_user.username, project.name)
    try:
        return project.export_raw(project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/generations", dependencies=[Depends(verified_user)])
async def export_generations(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    number: int = Query(),
) -> Response:
    """
    Export annotations
    """
    table = project.generations.get_generated(
        project_slug=project.name,
        username=current_user.username,
        n_elements=number,
    )

    if "error" in table:
        raise HTTPException(status_code=500, detail=table["error"])

    # join the text
    table = table.join(project.content["text"], on="index")

    # convert to payload
    output = StringIO()
    pd.DataFrame(table).to_csv(output, index=False)
    csv_data = output.getvalue()
    output.close()
    headers = {
        "Content-Disposition": 'attachment; filename="data.csv"',
        "Content-Type": "text/csv",
    }

    return Response(content=csv_data, media_type="text/csv", headers=headers)
