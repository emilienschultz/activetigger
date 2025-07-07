import io
from typing import Annotated

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
    LMInformationsModel,
    SimpleModelModel,
    SimpleModelOutModel,
    TextDatasetModel,
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
        orchestrator.log_action(current_user.username, "TRAIN MODEL: simplemodel", project.name)
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
) -> LMInformationsModel:
    """
    Get Bert parameters and statistics
    """
    try:
        statistics = project.languagemodels.get_informations(name)
        return statistics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/bert/predict", dependencies=[Depends(verified_user)])
async def predict(
    project: Annotated[Project, Depends(get_project)],
    current_user: Annotated[UserInDBModel, Depends(verified_user)],
    model_name: str,
    scheme: str,
    dataset: str = "all",
    batch_size: int = 32,
    external_dataset: TextDatasetModel | None = None,
) -> None:
    """
    Start prediction with a model
    """

    try:
        # get the data

        if dataset == "train":
            # data from the training dataset used for the model
            df = project.schemes.get_scheme_data(scheme=scheme, complete=True, kind=["train"])
            # ids_train = project.languagemodels.get_train_ids(model_name)
            # df = df.loc[ids_train]
            col_label = "labels"
            col_id = None
        elif dataset == "all":
            df = pd.DataFrame(project.features.get_column_raw("text", index="all"))
            df[project.params.col_id] = project.features.get_column_raw(
                project.params.col_id, index="all"
            )  # add original id
            col_label = None
            col_id = project.params.col_id
        elif dataset == "external":
            if external_dataset is None:
                raise HTTPException(status_code=400, detail="External dataset is missing")
            csv_buffer = io.StringIO(external_dataset.csv)
            df = pd.read_csv(
                csv_buffer,
            )
            df["text"] = df[external_dataset.text]
            df["index"] = df[external_dataset.id].apply(str)
            # if len(df[external_dataset.id].unique()) == len(df):
            #     df["index"] = df[external_dataset.id].apply(str)
            # else:
            #     df["index"] = ["external-" + str(i) for i in range(len(df))]
            df.set_index("index", inplace=True)
            df = df[["text"]].dropna()
            col_label = None
            col_id = None
            # raise HTTPException(status_code=500, detail="Not implemented yet")
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
            dataset="external",
            batch_size=batch_size,
        )
        orchestrator.log_action(current_user.username, f"PREDICT MODEL: {model_name}", project.name)
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
        project.languagemodels.start_testing_process(
            project_slug=project.name,
            name=model,
            user=current_user.username,
            df=df,
            col_text="text",
            col_labels="labels",
        )
        orchestrator.log_action(current_user.username, "PREDICT MODEL TEST", project.name)
        return None

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
    print(f"bertmodel : {bert}")
    try:
        # check in the size limit is not exceeded
        limit = orchestrator.users.get_storage_limit(current_user.username)
        if orchestrator.users.get_storage(current_user.username) > limit * 1000:
            raise HTTPException(
                status_code=500,
                detail=f"User storage limit exceeded ({limit} Gb), please delete some models in your projects",
            )

        # Check if there is no other competing processes : 1 active process by user
        if len(project.languagemodels.current_user_processes(current_user.username)) > 0:
            raise Exception(
                "User already has a process launched, please wait before launching another one"
            )

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
        df = df[df["labels"].isin(label_counts[label_counts >= bert.class_min_freq].index)]

        # remove class requested by the user
        if len(bert.exclude_labels) > 0:
            print(f"Excluding labels : {bert.exclude_labels}")
            print(len(df))
            df = df[~df["labels"].isin(bert.exclude_labels)]
            bert.name = f"{bert.name}_exclude_labels_"
            print(len(df))

        # balance the dataset based on the min class
        if bert.class_balance:
            min_freq = df["labels"].value_counts().sort_values().min()
            df = (
                df.groupby("labels")
                .apply(lambda x: x.sample(min_freq))
                .reset_index(level=0, drop=True)
            )

        # launch training process
        project.languagemodels.start_training_process(
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

    test_rights("modify project", current_user.username, project.name)

    # get BERT process for username
    try:
        if specific_user is not None:
            user = specific_user
        else:
            user = current_user.username

        p = project.get_process(["train_bert", "predict_bert"], user)
        if len(p) == 0:
            raise HTTPException(status_code=400, detail="No process found")

        # get id
        unique_id = p[0].unique_id
        # kill the process
        orchestrator.queue.kill(unique_id)
        # delete it in the database if it is a training
        if p[0].kind == "train_bert":
            project.db_manager.language_models_service.delete_model(project.name, p[0].model_name)
        orchestrator.log_action(current_user.username, "STOP MODEL TRAINING", project.name)
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
    test_rights("modify project", current_user.username, project.name)

    try:
        project.languagemodels.rename(former_name, new_name)
        orchestrator.log_action(
            current_user.username,
            f"INFO RENAME MODEL: {former_name} -> {new_name}",
            project.name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
