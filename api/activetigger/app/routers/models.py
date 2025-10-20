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
    SimpleModelInModel,
    SimpleModelOutModel,
    TextDatasetModel,
    UserInDBModel,
)
from activetigger.orchestrator import orchestrator
from activetigger.project import Project

router = APIRouter(tags=["models"])


@router.post("/models/simple/train", dependencies=[Depends(verified_user)])
async def train_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    simplemodel: SimpleModelInModel,
) -> None:
    """
    Compute simplemodel
    """
    try:
        project.train_simplemodel(simplemodel, current_user.username)
        orchestrator.log_action(current_user.username, "TRAIN SIMPLE MODEL", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/simple/retrain", dependencies=[Depends(verified_user)])
async def retrain_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    scheme: str,
    name: str,
) -> None:
    """
    Retrain simplemodel
    """
    try:
        project.retrain_simplemodel(name, scheme, current_user.username)
        orchestrator.log_action(current_user.username, f"RETRAIN SIMPLE MODEL {name}", project.name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/simple/delete", dependencies=[Depends(verified_user)])
async def delete_quickmodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> None:
    """
    Delete simplemodel
    """
    try:
        test_rights(ProjectAction.DELETE, current_user.username, project.name)
        project.simplemodels.delete(name)
        orchestrator.log_action(
            current_user.username, f"DELETE SIMPLE MODEL + FEATURES: {name}", project.name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/simplemodel", dependencies=[Depends(verified_user)])
async def get_simplemodel(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    name: str,
) -> SimpleModelOutModel | None:
    """
    Get available simplemodel by a name
    """
    try:
        sm = project.simplemodels.get(name)
        return SimpleModelOutModel(
            model=sm.name,
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
        elif kind == "simple":
            return project.simplemodels.get_informations(name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/predict", dependencies=[Depends(verified_user)])
async def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    scheme: str,
    kind: str,
    dataset: str = "all",
    batch_size: int = 32,
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model
    - simple or bert model
    - types of dataset
    Manage specific cases for prediction

    TODO : add external again

    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        datasets = None

        if kind not in ["simple", "bert"]:
            raise Exception(f"Model kind {kind} not recognized")

        # managing the perimeter of the prediction
        if dataset == "all":
            datasets = ["train"]
            if project.valid is not None:
                datasets.append("valid")
            if project.test is not None:
                datasets.append("test")
        elif dataset == "external":
            raise Exception("External dataset not yet implemented for bert models")
        else:
            raise Exception(f"Dataset {dataset} not recognized")

        # case for bert models
        if kind == "bert":
            df = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=datasets)
            project.languagemodels.start_predicting_process(
                project_slug=project.name,
                name=model_name,
                user=current_user.username,
                df=df,
                col_text="text",
                col_label="labels",
                col_id="id",
                col_datasets="dataset",
                dataset=dataset,
                batch_size=batch_size,
                statistics=datasets,
            )

        # case for simple models
        if kind == "simple":
            sm = project.simplemodels.get(model_name)
            if sm is None:
                raise Exception(f"Simple model {model_name} not found")
            df = project.features.get(sm.features, dataset=dataset, keep_dataset_column=True)
            cols_features = [col for col in df.columns if col != "dataset"]
            labels = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=datasets)
            df["labels"] = labels["labels"]

            # add the data for the labels
            project.simplemodels.start_predicting_process(
                name=model_name,
                username=current_user.username,
                df=df,
                dataset=dataset,
                col_dataset="dataset",
                cols_features=cols_features,
                col_label="labels",
                statistics=datasets,
            )

        orchestrator.log_action(
            current_user.username,
            f"PREDICT MODEL: {model_name} - {kind} DATASET: {dataset}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    scheme: str,
    dataset: str = "all",
    batch_size: int = 32,
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model for a specific dataset
    Manage specific cases for prediction
    TO DEPRECATE WHEN THE MODEL SIMPLE AND BERT WILL BE MERGED
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        # get the data
        if dataset == "train":
            status = "predicting"
            statistics = "outofsample"
            df = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["train"])
            col_label = "labels"
            col_id = None

        elif dataset == "all":
            status = "predicting"
            statistics = "outofsample"
            col_label = None
            col_id = project.params.col_id
            df = pd.DataFrame(project.features.get_column_raw("text", index="all"))
            # fix in the case where the id column is the row
            if project.params.col_id != "dataset_row_number":
                df[project.params.col_id] = project.features.get_column_raw(
                    project.params.col_id, index="all"
                )  # add original id
            else:
                df["dataset_row_number"] = df.index

        elif dataset == "external":
            status = "predicting"
            statistics = "outofsample"
            col_label = None
            col_id = None
            if external_dataset is None:
                raise HTTPException(status_code=400, detail="External dataset is missing")
            csv_buffer = io.StringIO(external_dataset.csv)
            df = pd.read_csv(
                csv_buffer,
            )
            df["text"] = df[external_dataset.text]
            df["index"] = df[external_dataset.id].apply(str)
            df.set_index("index", inplace=True)
            df = df[["text"]].dropna()

        elif dataset == "test":
            status = "predicting"
            statistics = "full"
            col_label = "labels"
            col_id = None
            if project.schemes.test is None:
                raise HTTPException(status_code=500, detail="No test dataset for this project")
            df = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["test"])
            if set(project.languagemodels.get_labels(model_name)) != set(
                df["labels"].dropna().unique()
            ):
                raise Exception("The testset and the model have different labels")
            project.languagemodels.clean_files_valid(model_name, dataset)

        elif dataset == "valid":
            status = "predicting"
            statistics = "full"
            col_label = "labels"
            col_id = None
            if project.schemes.valid is None:
                raise HTTPException(status_code=500, detail="No valid dataset for this project")
            df = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["valid"])
            if set(project.languagemodels.get_labels(model_name)) != set(
                df["labels"].dropna().unique()
            ):
                raise Exception("The valid set and the model have different labels")
            project.languagemodels.clean_files_valid(model_name, dataset)

        else:
            raise Exception(f"dataset {dataset} not found")

        # start process to predict
        project.languagemodels.start_predicting_process(
            project_slug=project.name,
            name=model_name,
            user=current_user.username,
            df=df,
            col_text="text",
            col_label=col_label,
            col_id=col_id,
            dataset=dataset,
            batch_size=batch_size,
            status=status,
            statistics=statistics,
        )
        orchestrator.log_action(
            current_user.username, f"PREDICT MODEL: {model_name} DATASET: {dataset}", project.name
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


@router.post("/models/bert/stop", dependencies=[Depends(verified_user)])
async def stop_bert(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    specific_user: str | None = None,
) -> None:
    """
    Stop user process
    """
    test_rights(ProjectAction.ADD, current_user.username, project.name)
    try:
        if specific_user is not None:
            user = specific_user
        else:
            user = current_user.username

        orchestrator.stop_user_processes(kind=["train_bert", "predict_bert"], username=user)
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
async def save_bert(
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
