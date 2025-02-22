from typing import Annotated, Any

import pandas as pd
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    BertModelModel,
    SimpleModelModel,
    SimpleModelOutModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["models"])


@router.post("/models/simplemodel", dependencies=[Depends(verified_user)])
async def post_simplemodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    simplemodel: SimpleModelModel,
) -> None:
    """
    Compute simplemodel
    """
    try:
        project.update_simplemodel(simplemodel, current_user.username)
        orchestrator.log_action(
            current_user.username, "INFO compute simplemodel", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
) -> SimpleModelOutModel | None:
    """
    Get available simplemodel for the project/user/scheme if any
    """
    try:
        return project.simplemodels.get(scheme, current_user.username)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/bert", dependencies=[Depends(verified_user)])
async def get_bert(
    project: Annotated[Project, Depends(get_project)], name: str
) -> dict[str, Any]:
    """
    Get Bert parameters and statistics
    """
    b = project.bertmodels.get(name, lazy=True)
    if b is None:
        raise HTTPException(status_code=400, detail="Bert model does not exist")
    data = b.informations()
    return data


@router.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    dataset: str = "all",
    batch_size: int = 32,
) -> None:
    """
    Start prediction with a model
    """
    # get the data
    if dataset == "train":
        df = project.content[["text"]]  # get data
    elif dataset == "all":
        r = project.features.get_column_raw("text", index="all")
        if "success" in r:
            df = pd.DataFrame(r["success"])
        else:
            raise Exception("Problem with full dataset")
    else:
        raise Exception(f"dataset {dataset} not found")

    # start process to predict
    try:
        project.bertmodels.start_predicting_process(
            project_slug=project.name,
            name=model_name,
            user=current_user.username,
            df=df,
            col_text="text",
            dataset=dataset,
            batch_size=batch_size,
        )
        orchestrator.log_action(
            current_user.username, f"INFO predict bert {model_name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    TODO : move the methods to specific class
    """
    try:
        # get data
        df = project.schemes.get_scheme_data(bert.scheme, complete=True)
        df = df[["text", "labels"]].dropna()

        # management for multilabels / dichotomize
        if bert.dichotomize is not None:
            df["labels"] = df["labels"].apply(
                lambda x: project.schemes.dichotomize(x, bert.dichotomize)
            )
            bert.name = f"{bert.name}_multilabel_on_{bert.dichotomize}"

        # remove class under the threshold
        label_counts = df["labels"].value_counts()
        df = df[
            df["labels"].isin(label_counts[label_counts >= bert.class_min_freq].index)
        ]

        # balance the dataset based on the min class
        if bert.class_balance:
            min_freq = df["labels"].value_counts().sort_values().min()
            df = (
                df.groupby("labels")
                .apply(lambda x: x.sample(min_freq))
                .reset_index(level=0, drop=True)
            )

        # launch training process
        project.bertmodels.start_training_process(
            name=bert.name,
            project=project.name,
            user=current_user.username,
            scheme=bert.scheme,
            df=df,
            col_text=df.columns[0],
            col_label=df.columns[1],
            base_model=bert.base_model,
            params=bert.params,
            test_size=bert.test_size,
        )
        orchestrator.log_action(
            current_user.username, f"INFO train bert {bert.name}", project.name
        )
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
) -> None:
    """
    Stop user process
    """
    # get BERT process for username
    try:
        p = project.get_process(["train_bert", "predict_bert"], current_user.username)
        if len(p) == 0:
            raise HTTPException(status_code=400, detail="No process found")
        # get id
        unique_id = p[0].unique_id
        # kill the process
        orchestrator.queue.kill(unique_id)
        # delete it in the database
        project.bertmodels.projects_service.delete_model(project.name, p[0].model.name)
        orchestrator.log_action(
            current_user.username, "INFO stop bert training", project.name
        )
        return None
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/test", dependencies=[Depends(verified_user)])
async def start_test(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    model: str,
) -> None:
    """
    Start testing the model on the test set
    """
    if project.schemes.test is None:
        raise HTTPException(status_code=500, detail="No test dataset for this project")

    try:
        # get data labels + text
        df = project.schemes.get_scheme_data(scheme, complete=True, kind=["test"])

        # launch testing process : prediction
        project.bertmodels.start_testing_process(
            project_slug=project.name,
            name=model,
            user=current_user.username,
            df=df,
            col_text="text",
            col_labels="labels",
        )
        orchestrator.log_action(
            current_user.username, "INFO predict bert for testing", project.name
        )
        return None

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/delete", dependencies=[Depends(verified_user)])
async def delete_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert_name: str,
) -> None:
    """
    Delete trained bert model
    """
    test_rights("modify project", current_user.username, project.name)

    try:
        project.bertmodels.delete(bert_name)
        orchestrator.log_action(
            current_user.username, f"INFO delete bert model {bert_name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/rename", dependencies=[Depends(verified_user)])
async def save_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename bertmodel
    """
    test_rights("modify project", current_user.username, project.name)

    try:
        project.bertmodels.rename(former_name, new_name)
        orchestrator.log_action(
            current_user.username,
            f"INFO rename bert model {former_name} - {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
