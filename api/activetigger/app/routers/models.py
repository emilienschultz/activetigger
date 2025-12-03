import io
from typing import Annotated

import pandas as pd  # type: ignore[import]
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
    check_storage,
    get_project,
    test_rights,
    verified_user,
)
from activetigger.datamodels import (
    BertModelModel,
    ModelInformationsModel,
    QuickModelInModel,
    QuickModelOutModel,
    TextDatasetModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["models"])


@router.post("/models/quick/train", dependencies=[Depends(verified_user)])
async def train_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    quickmodel: QuickModelInModel,
) -> None:
    """
    Compute quickmodel
    """
    try:
        project.train_quickmodel(quickmodel, current_user.username)
        orchestrator.log_action(
            current_user.username, f"TRAIN SIMPLE MODEL {quickmodel.name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/retrain", dependencies=[Depends(verified_user)])
async def retrain_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    name: str,
) -> None:
    """
    Retrain quickmodel
    """
    try:
        project.retrain_quickmodel(name, scheme, current_user.username)
        orchestrator.log_action(current_user.username, f"RETRAIN SIMPLE MODEL {name}", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/delete", dependencies=[Depends(verified_user)])
async def delete_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete quickmodel
    """
    try:
        test_rights(ProjectAction.DELETE, current_user.username, project.name)
        project.quickmodels.delete(name)
        orchestrator.log_action(
            current_user.username, f"DELETE SIMPLE MODEL + FEATURES: {name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/quick/rename", dependencies=[Depends(verified_user)])
async def save_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename quickmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.quickmodels.rename(former_name, new_name)
        orchestrator.log_action(
            current_user.username,
            f"INFO RENAME QUICK MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/quickmodel", dependencies=[Depends(verified_user)])
async def get_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> QuickModelOutModel | None:
    """
    Get available quickmodel by a name
    """
    try:
        sm = project.quickmodels.get(name)
        return QuickModelOutModel(
            name=sm.name,
            model=sm.model_type,
            params=sm.model_params,
            features=sm.features,
            statistics_train=sm.statistics_train,
            statistics_test=sm.statistics_test,
            statistics_cv10=sm.statistics_cv10,
            scheme=sm.scheme,
            username=sm.user,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/information", dependencies=[Depends(verified_user)])
async def get_bert(
    project: Annotated[Project, Depends(get_project)], name: str, kind: str
) -> ModelInformationsModel:
    """
    Get model information
    """
    try:
        if kind == "bert":
            return project.languagemodels.get_informations(name)
        elif kind == "quick":
            return project.quickmodels.get_informations(name)
        else:
            raise Exception(f"Model kind {kind} not recognized")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", dependencies=[Depends(verified_user)])
async def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    scheme: str,
    kind: str,
    dataset: str = "annotable",
    batch_size: int = 32,
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model
    - quick or bert model
    - types of dataset
    Manage specific cases for prediction

    TODO : optimize prediction on whole dataset
    TODO : manage prediction external/whole dataset for quick models

    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        datasets = None

        if kind not in ["quick", "bert"]:
            raise Exception(f"Model kind {kind} not recognized")

        # managing the perimeter of the prediction
        if dataset == "annotable":
            datasets = ["train"]
            if project.data.valid is not None:
                datasets.append("valid")
            if project.data.test is not None:
                datasets.append("test")
        elif dataset == "external":
            if kind != "bert":
                raise Exception("External dataset prediction is only available for bert models")
        elif dataset == "all":
            pass
        else:
            raise Exception(f"Dataset {dataset} not recognized")

        # case for bert models
        if kind == "bert":
            # case the prediction is done on an external dataset
            if dataset == "external":
                # TODO : load data in the job rather than in the api
                if external_dataset is None:
                    raise HTTPException(status_code=400, detail="External dataset is missing")
                # load the external dataset
                if not project.data.check_dataset_exists(external_dataset.filename):
                    raise HTTPException(
                        status_code=404,
                        detail=f"External dataset file {external_dataset.filename} not found",
                    )
                df = project.data.read_dataset(external_dataset.filename)
                df["text"] = df[external_dataset.text]
                df["index"] = df[external_dataset.id].apply(str)
                df["id"] = df["index"]
                df["dataset"] = "external"
                df.set_index("index", inplace=True)
                df = df[["id", "dataset", "text"]].dropna()
                col_label = None
                datasets = None
            # case the prediction is done on all the data
            elif dataset == "all":
                df = pd.DataFrame(project.features.get_column_raw("text", index="all"))
                if project.params.col_id != "dataset_row_number":
                    df["id"] = project.features.get_column_raw(project.params.col_id, index="all")
                else:
                    df["id"] = df.index
                df["dataset"] = "all"
                col_label = None
            # case the prediction is done on annotable data
            else:
                if datasets is None:
                    raise Exception("Datasets variable should be defined for annotable dataset")
                df = project.schemes.get_scheme(scheme=scheme, complete=True, kind=datasets)
                col_label = "labels"
            project.languagemodels.start_predicting_process(
                project_slug=project.name,
                name=model_name,
                user=current_user.username,
                df=df,
                col_text="text",
                col_label=col_label,
                col_id="id",
                col_datasets="dataset",
                dataset=dataset,
                batch_size=batch_size,
                statistics=datasets,
            )

        # case for quick models
        if kind == "quick":
            if datasets is None:
                raise Exception("Dataset parameter must be specified for quick model prediction")
            sm = project.quickmodels.get(model_name)
            if sm is None:
                raise Exception(f"Quick model {model_name} not found")

            # build the X, y dataframe
            df = project.features.get(sm.features, dataset=dataset, keep_dataset_column=True)
            cols_features = [col for col in df.columns if col != "dataset"]
            labels = project.schemes.get_scheme(scheme=scheme, complete=True, kind=datasets)
            df["labels"] = labels["labels"]
            df["text"] = labels["text"]

            # add the data for the labels
            project.quickmodels.start_predicting_process(
                name=model_name,
                username=current_user.username,
                df=df,
                dataset=dataset,
                col_dataset="dataset",
                cols_features=cols_features,
                col_label="labels",
                statistics=datasets,
                col_text="text",  # TODO : fx bug
            )

        orchestrator.log_action(
            current_user.username,
            f"PREDICT MODEL: {model_name} - {kind} DATASET: {dataset}",
            project.name,
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
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        check_storage(current_user.username)
        project.start_languagemodel_training(
            bert=bert,
            username=current_user.username,
        )
        orchestrator.log_action(current_user.username, f"TRAIN MODEL: {bert.name}", project.name)
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
    test_rights(ProjectAction.DELETE, current_user.username, project.name)
    try:
        # delete the model
        project.languagemodels.delete(bert_name)
        # delete the features associated with the model
        for f in [i for i in project.features.map.keys() if bert_name.replace("__", "_") in i]:
            project.features.delete(f)
        orchestrator.log_action(
            current_user.username, f"DELETE MODEL + FEATURES: {bert_name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/rename", dependencies=[Depends(verified_user)])
async def rename_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    former_name: str,
    new_name: str,
) -> None:
    """
    Rename bertmodel
    """
    test_rights(ProjectAction.UPDATE, current_user.username, project.name)
    try:
        project.languagemodels.rename(former_name, new_name)
        orchestrator.log_action(
            current_user.username,
            f"INFO RENAME MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
