import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
)

from activetigger.app.dependencies import (
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    ActionModel,
    AnnotationModel,
    AnnotationsDataModel,
    ElementOutModel,
    NextInModel,
    ProjectionInStrictModel,
    ProjectionOutModel,
    ReconciliationModel,
    TableAnnotationsModel,
    TableOutModel,
    UserInDBModel,
    WaitingModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

logger = logging.getLogger(__name__)

# declare router
router = APIRouter()


@router.post("/elements/next", dependencies=[Depends(verified_user)])
async def get_next(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    next: NextInModel,
) -> ElementOutModel:
    """
    Get next element
    """
    try:
        r = project.get_next(
            scheme=next.scheme,
            selection=next.selection,
            sample=next.sample,
            user=current_user.username,
            label=next.label,
            history=next.history,
            frame=next.frame,
            filter=next.filter,
            dataset=next.dataset,
        )
        return ElementOutModel(**r)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/elements/projection", dependencies=[Depends(verified_user)])
async def get_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
) -> ProjectionOutModel | None:
    """
    Get projection if computed
    """

    try:
        # get the projection
        projection = project.projections.get(current_user.username)
        if projection is None:
            return None

        # add existing annotations
        df = project.schemes.get_scheme_data(scheme, complete=True)
        projection.labels = list(df["labels"].fillna("NA"))

        # add predictions if available
        if current_user.username in project.simplemodels.existing:
            if scheme in project.simplemodels.existing[current_user.username]:
                projection.predictions = list(
                    project.simplemodels.existing[current_user.username][scheme].proba["prediction"]
                )

        return projection
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/projection/compute", dependencies=[Depends(verified_user)])
async def compute_projection(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    projection: ProjectionInStrictModel,
) -> WaitingModel:
    """
    Start projection computation using futures
    Dedicated process, end with a file on the project
    projection__user.parquet
    """
    if len(projection.features) == 0:
        raise HTTPException(status_code=400, detail="No feature available")
    try:
        # get features from project
        features = project.features.get(projection.features)
        # compute the projection
        project.projections.compute(project.name, current_user.username, projection, features)
        orchestrator.log_action(
            current_user.username,
            f"COMPUTE PROJECTION: {projection.method}",
            project.params.project_slug,
        )
        return WaitingModel(detail=f"Projection {projection.method} is computing")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/elements/table", dependencies=[Depends(verified_user)])
async def get_list_elements(
    project: Annotated[Project, Depends(get_project)],
    scheme: str,
    min: int = 0,
    max: int = 0,
    contains: str | None = None,
    mode: str = "all",
    dataset: str = "train",
) -> TableOutModel:
    """
    Get a table of elements
    """
    try:
        print(
            f"Getting table for scheme {scheme} with min={min}, max={max}, mode={mode}, contains={contains}, dataset={dataset}"
        )
        extract = project.schemes.get_table(scheme, min, max, mode, contains, dataset)
        df = extract.batch.fillna(" ")
        table = (df.reset_index()[["id", "timestamp", "labels", "text", "comment"]]).to_dict(
            orient="records"
        )
        return TableOutModel(
            items=table,
            total=extract.total,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/table", dependencies=[Depends(verified_user)])
async def post_list_elements(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    table: TableAnnotationsModel,
) -> None:
    """
    Update a table of annotations
    """
    errors = []
    # loop on annotations
    for annotation in table.annotations:
        if annotation.label is None or annotation.element_id is None:
            errors.append(annotation)
            continue

        try:
            project.schemes.push_annotation(
                annotation.element_id,
                annotation.label,
                annotation.scheme,
                current_user.username,
                table.dataset,
                "table",
            )
            orchestrator.log_action(
                current_user.username,
                f"UPDATE ANNOTATION: in {annotation.scheme} element {annotation.element_id} as {annotation.label}",
                project.name,
            )
        except Exception:
            errors.append(annotation)
            continue

    if len(errors) > 0:
        raise HTTPException(
            status_code=500,
            detail="Error with some of the annotations - " + str(errors),
        )

    return None


@router.post("/annotation/file", dependencies=[Depends(verified_user)])
async def post_annotation_file(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    annotationsdata: AnnotationsDataModel,
) -> None:
    """
    Load annotations file
    """
    try:
        project.schemes.add_file_annotations(
            annotationsdata=annotationsdata, user=current_user.username, dataset="train"
        )
        orchestrator.log_action(
            current_user.username,
            f"LOAD ANNOTATION FROM FILE: scheme {annotationsdata.scheme}",
            project.name,
        )
        return None
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/elements/{element_id}", dependencies=[Depends(verified_user)])
async def get_element(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    element_id: str,
    scheme: str,
    dataset: str = "train",
) -> ElementOutModel:
    """
    Get specific element
    """
    try:
        r = project.get_element(
            element_id, scheme=scheme, user=current_user.username, dataset=dataset
        )
        return ElementOutModel(**r)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/annotation/{action}", dependencies=[Depends(verified_user)])
async def post_annotation(
    action: ActionModel,
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    annotation: AnnotationModel,
) -> None:
    """
    Add, Update, Delete annotations
    Comment :
    - For the moment add == update
    - No information kept of selection process
    """

    # manage rights
    test_rights("modify annotation", current_user.username, project.name)

    if action in ["add", "update"]:
        try:
            project.schemes.push_annotation(
                annotation.element_id,
                annotation.label,
                annotation.scheme,
                current_user.username,
                annotation.dataset,
                annotation.comment,
                annotation.selection,
            )

            orchestrator.log_action(
                current_user.username,
                f"ADD ANNOTATION: in {annotation.scheme} element {annotation.element_id} as {annotation.label} ({annotation.dataset})",
                project.name,
            )
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    if action == "delete":
        try:
            project.schemes.delete_annotation(
                annotation.element_id,
                annotation.scheme,
                annotation.dataset,
                current_user.username,
            )

            orchestrator.log_action(
                current_user.username,
                f"DELETE ANNOTATION: in {annotation.scheme} id {annotation.element_id}",
                project.name,
            )
            return None
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    raise HTTPException(status_code=400, detail="Wrong action")


@router.get("/elements/reconciliate", dependencies=[Depends(verified_user)])
async def get_reconciliation_table(
    project: Annotated[Project, Depends(get_project)], scheme: str
) -> ReconciliationModel:
    """
    Get the reconciliation table
    """
    try:
        df, users = project.schemes.get_reconciliation_table(scheme)
        return ReconciliationModel(table=df.to_dict(orient="records"), users=users)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/elements/reconciliate", dependencies=[Depends(verified_user)])
async def post_reconciliation(
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    project: Annotated[Project, Depends(get_project)],
    users: list = Query(),
    element_id: str = Query(),
    label: str = Query(),
    scheme: str = Query(),
) -> None:
    """
    Post a label for all user in a list
    """

    try:
        # for each user
        for u in users:
            project.schemes.push_annotation(element_id, label, scheme, u, "train", "reconciliation")

        # add a new tag for the reconciliator
        project.schemes.push_annotation(
            element_id,
            label,
            scheme,
            current_user.username,
            "reconciliation",
            "reconciliation",
        )

        # log
        orchestrator.log_action(
            current_user.username,
            f"RECONCILIATE ANNOTATION: in {scheme} element {element_id} as {label}",
            project.name,
        )
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
