import asyncio
from typing import Annotated

import pandas as pd  # type: ignore[import]
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
)

from activetigger.app.dependencies import (
    ProjectAction,
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

    def _impl():
        test_rights(ProjectAction.ADD, current_user.username, project.name)
        try:
            project.train_quickmodel(quickmodel, current_user.username)
            orchestrator.log_action(
                current_user.username, f"TRAIN SIMPLE MODEL {quickmodel.name}", project.name
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


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

    def _impl():
        test_rights(ProjectAction.GET, current_user.username, project.name)
        try:
            project.retrain_quickmodel(name, scheme, current_user.username)
            orchestrator.log_action(current_user.username, f"RETRAIN SIMPLE MODEL {name}", project.name)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.post("/models/quick/delete", dependencies=[Depends(verified_user)])
async def delete_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete quickmodel
    """

    def _impl():
        test_rights(ProjectAction.DELETE, current_user.username, project.name)
        try:
            project.quickmodels.delete(name)
            orchestrator.log_action(
                current_user.username, f"DELETE SIMPLE MODEL + FEATURES: {name}", project.name
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


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

    def _impl():
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

    return await asyncio.to_thread(_impl)


@router.get("/models/quick", dependencies=[Depends(verified_user)])
async def get_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> QuickModelOutModel | None:
    """
    Get available quickmodel by a name
    """

    def _impl():
        test_rights(ProjectAction.GET, current_user.username, project.name)
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
                balance_classes=sm.balance_classes,
                scheme=sm.scheme,
                username=sm.user,
                exclude_labels=sm.exclude_labels if "exclude_labels" in sm else []
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.get("/models/information", dependencies=[Depends(verified_user)])
async def get_model_information(
    project: Annotated[Project, Depends(get_project)], name: str, kind: str
) -> ModelInformationsModel:
    """
    Get model information
    """

    def _impl():
        try:
            if kind == "bert":
                return project.languagemodels.get_informations(name)
            elif kind == "quick":
                return project.quickmodels.get_informations(name)
            else:
                raise Exception(f"Model kind {kind} not recognized")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


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

    def _impl():
        test_rights(ProjectAction.ADD, current_user.username, project.name)
        try:
            # types of prediction
            if kind not in ["quick", "bert"]:
                raise Exception(f"Model kind {kind} not recognized")

            if dataset not in ["annotable", "external", "all"]:
                raise Exception(f"Dataset {dataset} not recognized")

            # managing the perimeter of the prediction
            datasets = None
            if dataset == "annotable":
                datasets = ["train"]
                if project.data.valid is not None:
                    datasets.append("valid")
                if project.data.test is not None:
                    datasets.append("test")
            if dataset == "external":
                if kind != "bert":
                    raise Exception("External dataset prediction is only available for bert models")

            # case for bert models
            if kind == "bert":
                # case the prediction is done on an external dataset
                if dataset == "external":
                    if external_dataset is None:
                        raise Exception("External dataset must be provided for external prediction")
                    if not project.data.get_path(external_dataset.filename).exists():
                        raise HTTPException(
                            status_code=404,
                            detail=f"External dataset file {external_dataset.filename} not found",
                        )
                    df = None
                    col_label = None
                    datasets = None
                    path_data = project.data.get_path(external_dataset.filename)

                # case the prediction is done on all the data
                elif dataset == "all":
                    df = None
                    col_label = None
                    datasets = None
                    path_data = project.data.path_data_all

                # case the prediction is done on annotable data
                else:
                    if datasets is None:
                        raise Exception("No dataset available for prediction")
                    df = project.schemes.get_scheme(
                        scheme=scheme, complete=True, datasets=datasets, id_external=True
                    )
                    col_label = "labels"
                    path_data = None
                    print(df)

                project.languagemodels.start_predicting_process(
                    project_slug=project.name,
                    name=model_name,
                    user=current_user.username,
                    df=df,
                    col_label=col_label,
                    dataset=dataset,
                    batch_size=batch_size,
                    statistics=datasets,
                    path_data=path_data,
                    external_dataset=external_dataset,
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
                labels = project.schemes.get_scheme(scheme=scheme, complete=True, datasets=datasets)
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
                    col_text="text",
                )

            orchestrator.log_action(
                current_user.username,
                f"PREDICT MODEL: {model_name} - {kind} DATASET: {dataset}",
                project.name,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.post("/models/bert/train", dependencies=[Depends(verified_user)])
async def post_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert: BertModelModel,
) -> None:
    """
    Compute bertmodel
    """

    def _impl():
        test_rights(ProjectAction.ADD, current_user.username, project.name)
        try:
            if not orchestrator.available_storage(current_user.username):
                raise HTTPException(
                    status_code=403,
                    detail="Storage limit exceeded. Please delete models orcontact the administrator.",
                )
            project.start_languagemodel_training(
                bert=bert,
                username=current_user.username,
            )
            orchestrator.log_action(current_user.username, f"TRAIN MODEL: {bert.name}", project.name)
            return None

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await asyncio.to_thread(_impl)


@router.post("/models/bert/delete", dependencies=[Depends(verified_user)])
async def delete_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    bert_name: str,
) -> None:
    """
    Delete trained bert model
    # TODO : check the replace
    """

    def _impl():
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

    return await asyncio.to_thread(_impl)


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

    def _impl():
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

    return await asyncio.to_thread(_impl)
