import json
import os
import pickle
import shutil
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame
from pydantic import ValidationError
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import activetigger.functions as functions
from activetigger.datamodels import (
    BertParams,
    KnnParams,
    LassoParams,
    LiblinearParams,
    Multi_naivebayesParams,
    RandomforestParams,
)


class BertModel:
    """
    Manage one bertmodel
    """

    name: str
    path: Path
    params: dict | None
    base_model: str | None
    tokenizer = None
    model = None
    log_history = None
    status: str
    pred: DataFrame | None
    data: DataFrame | None
    timestamp: datetime

    def __init__(
        self,
        name: str,
        path: Path,
        base_model: str | None = None,
        params: dict | None = None,
    ) -> None:
        """
        Init a bert model
        """
        self.name = name
        self.path = path
        self.params = params
        self.base_model = base_model
        self.tokenizer = None
        self.model = None
        self.log_history = None
        self.status = "initializing"
        self.pred = None
        self.data = None
        self.timestamp = datetime.now()

    def __repr__(self) -> str:
        return f"{self.name} - {self.base_model}"

    def load(self, lazy=False):
        """
        Load trained model from files
        - either lazy (only parameters)
        - or complete (the weights of the model)
        """
        if not (self.path / "config.json").exists():
            raise FileNotFoundError("model not defined")

        # Load parameters
        with open(self.path / "parameters.json", "r") as jsonfile:
            self.params = json.load(jsonfile)

        # Load training data
        self.data = pd.read_parquet(self.path / "training_data.parquet")

        # Load train history
        with open(self.path / "log_history.txt", "r") as f:
            self.log_history = json.load(f)

        # Load prediction if available
        if (self.path / "predict.parquet").exists():
            self.pred = pd.read_parquet(self.path / "predict.parquet")

        # Only load the model if not lazy mode
        if lazy:
            self.status = "lazy"
        else:
            with open(self.path / "config.json", "r") as jsonfile:
                modeltype = json.load(jsonfile)["_name_or_path"]
            self.tokenizer = AutoTokenizer.from_pretrained(modeltype)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.path)
            self.status = "loaded"

    def informations(self) -> dict:
        """
        Compute statistics for train & test
        - load statistics if computed
        - update them if possible
            - only training information
            - train scores
            - test scores
        """
        flag_modification = False
        if (self.path / "statistics.json").exists():
            with open(self.path / "statistics.json", "r") as f:
                r = json.load(f)
        else:
            r = {}

        # all informations already computed
        if len(r) == 3:
            return r

        # add training informations
        if not "training" in r:
            log = self.log_history
            loss = pd.DataFrame(
                [
                    [
                        log[2 * i]["epoch"],
                        log[2 * i]["loss"],
                        log[2 * i + 1]["eval_loss"],
                    ]
                    for i in range(0, int((len(log) - 1) / 2))
                ],
                columns=["epoch", "val_loss", "val_eval_loss"],
            )
            r["training"] = {
                "loss": loss.to_json(orient="columns"),
                "parameters": self.params,
            }
            flag_modification = True

        # add train scores
        if (not "train_scores" in r) and (self.path / "predict.parquet").exists():
            df = self.data.copy()
            df["prediction"] = self.pred["prediction"]
            Y_pred = df["prediction"]
            Y = df["labels"]
            f = df.apply(lambda x: x["prediction"] != x["labels"], axis=1)
            r["train_scores"] = {
                "f1_micro": f1_score(Y, Y_pred, average="micro"),
                "f1_macro": f1_score(Y, Y_pred, average="macro"),
                "f1_weighted": f1_score(Y, Y_pred, average="weighted"),
                "f1": list(f1_score(Y, Y_pred, average=None)),
                "precision": list(precision_score(list(Y), list(Y_pred), average=None)),
                "recall": list(recall_score(list(Y), list(Y_pred), average=None)),
                "accuracy": accuracy_score(Y, Y_pred),
                "false_prediction": df[f][["text", "labels", "prediction"]]
                .reset_index()
                .to_json(orient="records"),
            }
            flag_modification = True

        # add test scores
        if (not "test_scores" in r) and (self.path / "predict_test.parquet").exists():
            df = pd.read_parquet(self.path / "predict_test.parquet")[
                ["prediction", "labels"]
            ].dropna()
            Y_pred = df["prediction"]
            Y = df["labels"]
            f = df.apply(lambda x: x["prediction"] != x["labels"], axis=1)
            r["test_scores"] = {
                "f1_micro": f1_score(Y, Y_pred, average="micro"),
                "f1_macro": f1_score(Y, Y_pred, average="macro"),
                "f1_weighted": f1_score(Y, Y_pred, average="weighted"),
                "f1": list(f1_score(Y, Y_pred, average=None)),
                "precision": precision_score(list(Y), list(Y_pred), average="micro"),
                "recall": list(recall_score(list(Y), list(Y_pred), average=None)),
                "accuracy": accuracy_score(Y, Y_pred),
            }
            flag_modification = True

        # if modifications
        if flag_modification:
            with open(self.path / "statistics.json", "w") as f:
                json.dump(r, f)
        return r


class BertModels:
    """
    Managing bertmodel training

    Comments:
        All the data are sorted in path/bert/$NAME

    TODO : std.err in the logs for processes
    """

    queue: Any
    path: Path
    computing: dict

    def __init__(self, path: Path, queue: Any) -> None:
        self.params_default = {
            "batchsize": 4,
            "gradacc": 1,
            "epochs": 3,
            "lrate": 5e-05,
            "wdecay": 0.01,
            "best": True,
            "eval": 10,
            "gpu": False,
            "adapt": True,
        }
        self.base_models = [
            "camembert/camembert-base",
            "camembert/camembert-large",
            "flaubert/flaubert_small_cased",
            "flaubert/flaubert_base_cased",
            "flaubert/flaubert_large_cased",
            "distilbert-base-cased",
            "roberta-base",
            "microsoft/deberta-base",
            "distilbert-base-multilingual-cased",
            "microsoft/Multilingual-MiniLM-L12-H384",
            "xlm-roberta-base",
        ]
        self.queue = queue
        self.path: Path = Path(path) / "bert"
        if not self.path.exists():
            os.mkdir(self.path)

        # keep current processes (one by user max)
        self.computing: dict = {}

    def __repr__(self) -> str:
        return f"Trained models : {self.trained()}"

    def trained(self) -> dict:
        """
        Trained bert by scheme in the project
        + if prediction available
        + compression if available / launch it
        """
        r: dict = {}
        if self.path.exists():  # if bert models have been trained
            all_files = os.listdir(self.path)
            trained = [
                i
                for i in all_files
                if os.path.isdir(self.path / i)
                and (self.path / i / "finished").exists()
            ]
            for i in trained:
                predict = False
                compressed = False
                # test if prediction available
                if (self.path / i / "predict.parquet").exists():
                    predict = True
                # test if compression available
                if (self.path / "../../static" / f"{i}.tar.gz").exists():
                    compressed = True
                else:
                    self.start_compression(i)
                scheme = i.split("__")[-1]  # scheme after __
                if not scheme in r:
                    r[scheme] = {}
                r[scheme][i] = {"predicted": predict, "compressed": compressed}
        return r

    def training(self) -> dict:
        """
        Currently under training
        """
        return {u: self.computing[u][0].name for u in self.computing}

    def delete(self, bert_name: str) -> dict:
        """
        Delete bert model
        """
        if not (self.path / bert_name).exists():
            return {"error": "Bert model does not exist"}

        try:
            shutil.rmtree(self.path / bert_name)
            os.remove(self.path / "../../static" / f"{bert_name}.tar.gz")
            return {"success": "Bert model deleted"}
        except:
            return {"error": "An error occured in deleting bert model"}

    def start_training_process(
        self,
        name: str,
        user: str,
        scheme: str,
        df: DataFrame,
        col_text: str,
        col_label: str,
        base_model: str | None = None,
        params: dict | None = None,
        test_size: float | None = None,
    ) -> dict:
        """
        Manage the training of a model from the API
        """

        # Check if there is no other competing processes
        # For the moment : 1 active process by user
        if user in self.computing:
            return {"error": "processes already launched, cancel it before"}

        # set default parameters if needed
        if base_model is None:
            base_model = "almanach/camembert-base"
        if params is None:
            params = self.params_default
        if test_size is None:
            test_size = 0.2
        # test json parameters
        try:
            e = BertParams(**params)
        except ValidationError as e:
            return {"error": e.json()}

        # name integrating the scheme & user + date
        current_date = datetime.now()
        day = current_date.strftime("%d")
        month = current_date.strftime("%m")
        year = current_date.strftime("%Y")
        name = f"{name}__{user}__{day}-{month}-{year}__{scheme}"

        # launch as a independant process
        args = {
            "path": self.path,
            "name": name,
            "df": df,
            "col_label": col_label,
            "col_text": col_text,
            "base_model": base_model,
            "params": params,
            "test_size": test_size,
        }

        unique_id = self.queue.add("training", functions.train_bert, args)

        # Update the queue
        b = BertModel(name, self.path / name, base_model)
        b.status = "training"
        self.computing[user] = [b, unique_id]

        return {"success": "bert model on training"}

    def start_testing_process(
        self, name: str, user: str, df: DataFrame, col_text: str, col_labels: str
    ):
        """
        Start testing process
        - launch as an independant process functions.test_bert
        - once computed, sync with the queue
        """
        if user in self.computing:
            return {"error": "Processes already launched, cancel it before"}

        if not (self.path / name).exists():
            return {"error": "This model does not exist"}

        # delete previous files
        if (self.path / "predict_test.parquet").exists():
            os.remove(self.path / "predict_test.parquet")
        if (self.path / "statistics.json").exists():
            os.remove(self.path / "statistics.json")

        # start prediction on the test set
        b = BertModel(name, self.path / name)
        b.load()
        args = {
            "df": df,
            "col_text": col_text,
            "col_labels": col_labels,
            "model": b.model,
            "tokenizer": b.tokenizer,
            "path": b.path,
            "file_name": "predict_test.parquet",
        }
        unique_id = self.queue.add("prediction", functions.predict_bert, args)
        b.status = "testing"
        self.computing[user] = [b, unique_id]

        return {"success": "bert testing predicting"}

    def start_predicting_process(
        self, name: str, user: str, df: DataFrame, col_text: str
    ):
        """
        Start predicting process
        """
        if user in self.computing:
            return {"error": "Processes already launched, cancel it before"}

        if not (self.path / name).exists():
            return {"error": "This model does not exist"}

        b = BertModel(name, self.path / name)
        b.load()
        args = {
            "df": df,
            "col_text": col_text,
            "model": b.model,
            "tokenizer": b.tokenizer,
            "path": b.path,
            "file_name": "predict.parquet",
        }
        unique_id = self.queue.add("prediction", functions.predict_bert, args)
        b.status = "predicting"
        self.computing[user] = [b, unique_id]
        return {"success": "bert model predicting"}

    def start_compression(self, name):
        """
        Compress bertmodel as a separate process
        """
        process = Process(
            target=shutil.make_archive,
            args=(self.path / "../../static" / name, "gztar", self.path / name),
        )
        process.start()
        print("starting compression")

    def stop_user_process(self, user: str):
        """
        Stop the process of an user
        """
        if not user in self.computing:
            return {"error": "no current processes"}
        self.computing[user][1].terminate()  # end process

        # delete files in case of training
        b = self.computing[user][0]
        if b.status == "training":
            shutil.rmtree(self.computing[user][0].path)
        del self.computing[user]  # delete process
        return {"success": "process terminated"}

    def rename(self, former_name: str, new_name: str):
        """
        Rename a model (copy it)
        """
        if not (self.path / former_name).exists():
            return {"error": "no model currently trained"}
        if (self.path / new_name).exists():
            return {"error": "this name already exists"}
        if (self.path / former_name / "status.log").exists():
            return {"error": "model not trained completly"}

        # keep the scheme information
        if not "__" in new_name:
            new_name = new_name + "__" + former_name.split("__")[-1]

        os.rename(self.path / former_name, self.path / new_name)
        return {"success": "model renamed"}

    def get(self, name: str, lazy=False) -> BertModel | None:
        """
        Get a model
        """
        if not (self.path / name).exists():
            return None
        if (self.path / name / "status.log").exists():
            # process not finished
            return None
        b = BertModel(name, self.path / name)
        b.load(lazy=lazy)
        return b

    def update_processes(self) -> dict:
        """
        Update current computing
        Return features to add
        """
        predictions = {}
        for u in self.computing.copy():
            unique_id = self.computing[u][1]
            # case the process have been canceled, clean
            if not unique_id in self.queue.current:
                del self.computing[u]
                continue

            # else check its state
            if self.queue.current[unique_id]["future"].done():
                b = self.computing[u][0]
                if b.status == "predicting":
                    print("Prediction finished")
                    df = self.queue.current[unique_id]["future"].result()
                    predictions["predict_" + b.name] = df["prediction"]
                if b.status == "training":
                    print("Model trained")
                if b.status == "testing":
                    df = self.queue.current[unique_id]["future"].result()
                    print("Model tested")
                del self.computing[u]
                self.queue.delete(unique_id)
        return predictions

    def export_prediction(self, name: str, format: str | None = None):
        """
        Export predict file if exists
        """
        file_name = f"predict.parquet"
        path = self.path / name / file_name

        # change format if needed
        if format == "csv":
            df = pd.read_parquet(path)
            print(df)
            file_name = f"predict.csv"
            path = self.path / name / file_name
            df.to_csv(path)

        print(path)
        if not path.exists():
            return {"error": "file does not exist"}

        r = {"name": file_name, "path": path}
        return r

    def export_bert(self, name: str):
        """
        Export bert archive if exists
        """
        file_name = f"{name}.tar.gz"
        if not (self.path / "../../static" / file_name).exists():
            return {"error": "file does not exist"}
        r = {"name": file_name, "path": self.path / "../../static" / file_name}
        return r


class SimpleModels:
    """
    Managing simplemodels
    - define available models
    - save a simplemodel/user
    - train simplemodels
    """

    available_models: dict
    validation: dict
    existing: dict
    computing: dict
    path: Path
    queue: Any
    save_file: str

    def __init__(self, path: Path, queue):
        """
        Init Simplemodels class
        """
        # Models and default parameters
        self.available_models = {
            "liblinear": {"cost": 1},
            "knn": {"n_neighbors": 3},
            "randomforest": {"n_estimators": 500, "max_features": None},
            "lasso": {"C": 32},
            "multi_naivebayes": {"alpha": 1, "fit_prior": True, "class_prior": None},
        }

        # To validate JSON
        self.validation = {
            "liblinear": LiblinearParams,
            "knn": KnnParams,
            "randomforest": RandomforestParams,
            "lasso": LassoParams,
            "multi_naivebayes": Multi_naivebayesParams,
        }
        self.existing: dict = {}  # computed simplemodels
        self.computing: dict = {}  # curently under computation
        self.path: Path = path  # path to operate
        self.queue = queue  # access to executor for multiprocessing
        self.save_file: str = "simplemodels.pickle"  # file to save current state
        self.loads()  # load existing simplemodels

    def __repr__(self) -> str:
        return str(self.available())

    def available(self):
        """
        Available simplemodels
        """
        r = {}
        for u in self.existing:
            r[u] = {}
            for s in self.existing[u]:
                sm = self.existing[u][s]
                r[u][s] = {
                    "model": sm.name,
                    "params": sm.model_params,
                    "features": sm.features,
                    "statistics": sm.statistics,
                }
        return r

    def training(self):
        """
        Training simplemodels
        """
        r = {}
        for u in self.computing:
            r[u] = list(self.computing[u].keys())
        return r

    def exists(self, user: str, scheme: str):
        """
        Test if a simplemodel exists for a user/scheme
        """
        if user in self.existing:
            if scheme in self.existing[user]:
                return True
        return False

    def get_model(self, user: str, scheme: str):
        """
        Select a specific model in the repo
        """
        if user not in self.existing:
            return "This user has no model"
        if scheme not in self.existing[user]:
            return "The model for this scheme does not exist"
        return self.existing[user][scheme]

    def load_data(self, data, col_label, col_predictors, standardize):
        """
        Load data
        """
        f_na = data[col_predictors].isna().sum(axis=1) > 0
        if f_na.sum() > 0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize X data
        if standardize:
            df_pred = self.standardize(data[~f_na][col_predictors])
        else:
            df_pred = data[~f_na][col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label], df_pred], axis=1)

        # data for training
        Y = df[col_label]
        X = df[col_predictors]
        labels = Y.unique()

        return X, Y, labels

    def standardize(self, df):
        """
        Apply standardization
        """
        scaler = StandardScaler()
        df_stand = scaler.fit_transform(df)
        return pd.DataFrame(df_stand, columns=df.columns, index=df.index)

    def add_simplemodel(
        self,
        user,
        scheme,
        features,
        name,
        df,
        col_labels,
        col_features,
        standardize,
        model_params: dict | None = None,
    ):
        """
        A a new simplemodel for a user and a scheme
        """
        X, Y, labels = self.load_data(df, col_labels, col_features, standardize)

        # default parameters
        if model_params is None:
            model_params = self.available_models[name]

        # Select model
        if name == "knn":
            model = KNeighborsClassifier(n_neighbors=int(model_params["n_neighbors"]))

        if name == "lasso":
            model = LogisticRegression(
                penalty="l1", solver="liblinear", C=model_params["C"]
            )

        if name == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            model = LogisticRegression(
                penalty="l2", solver="lbfgs", C=model_params["cost"]
            )

        if name == "randomforest":
            # params  Num. trees mtry  Sample fraction
            # Number of variables randomly sampled as candidates at each split:
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            model = RandomForestClassifier(
                n_estimators=model_params["n_estimators"],
                random_state=42,
                max_features=model_params["max_features"],
            )

        if name == "multi_naivebayes":
            # Only with dtf or tfidf for features
            # TODO: calculate class prior for docfreq & termfreq
            model = MultinomialNB(
                alpha=model_params["alpha"],
                fit_prior=model_params["fit_prior"],
                class_prior=model_params["class_prior"],
            )

        # launch the compuation (model + statistics) as a future process
        # TODO: refactore the SimpleModel class / move to API the executor call ?
        args = {"model": model, "X": X, "Y": Y, "labels": labels}
        print(args)
        unique_id = self.queue.add("simplemodel", functions.fit_model, args)
        # future_result = self.executor.submit(functions.fit_model, model=model, X=X, Y=Y, labels=labels)
        sm = SimpleModel(
            name, user, X, Y, labels, "computing", features, standardize, model_params
        )
        if user not in self.computing:
            self.computing[user] = {}
        self.computing[user][scheme] = {"queue": unique_id, "sm": sm}

    def dumps(self):
        """
        Dumps all simplemodels to a pickle
        """
        with open(self.path / self.save_file, "wb") as file:
            pickle.dump(self.existing, file)

    def loads(self) -> bool:
        """
        Load all simplemodels from a pickle
        """
        if not (self.path / self.save_file).exists():
            return False
        with open(self.path / self.save_file, "rb") as file:
            self.existing = pickle.load(file)
        return True

    def update_processes(self):
        """
        Update current computing simplemodels
        """
        for u in self.computing.copy():
            s = list(self.computing[u].keys())[0]
            unique_id = self.computing[u][s]["queue"]
            if self.queue.current[unique_id]["future"].done():
                # TODO : deal better exception in the training
                try:
                    results = self.queue.current[unique_id]["future"].result()
                    sm = self.computing[u][s]["sm"]
                    sm.model = results["model"]
                    sm.proba = results["proba"]
                    sm.cv10 = results["cv10"]
                    sm.statistics = results["statistics"]
                    if u not in self.existing:
                        self.existing[u] = {}
                    self.existing[u][s] = sm
                    del self.computing[u]
                    self.queue.delete(unique_id)
                    self.dumps()
                except Exception as e:
                    print("Simplemodel failed")
                    print(e)
                    del self.computing[u]
                    self.queue.delete(unique_id)


class SimpleModel:
    name: str
    user: str
    features: list
    X: DataFrame
    Y: DataFrame
    labels: list
    model_params: dict
    standardize: bool
    proba: DataFrame
    statistics: dict
    cv10: DataFrame
    # model

    def __init__(
        self,
        name: str,
        user: str,
        X: DataFrame,
        Y: DataFrame,
        labels: list,
        model,
        features: list,
        standardize: bool,
        model_params: dict | None,
    ) -> None:
        """
        Define a specific Simplemodel with parameters
        TODO : add timestamp ?
        TODO : not sure that statistics function are still usefull since it is calculated during the fit
        """
        self.name = name
        self.user = user
        self.features = features
        self.X = X
        self.Y = Y
        self.labels = labels
        self.model_params = model_params
        self.standardize = standardize
        self.model = model
        self.proba = None
        self.statistics = None
        self.cv10 = None
        if not type(model) is str:  # TODO : tester si c'est un modèle
            self.proba = self.compute_proba(model, X)
            self.statistics = self.compute_statistics(model, X, Y, labels)
            self.cv10 = self.compute_10cv(model, X, Y)

    def json(self):
        """
        Return json representation
        """
        return {
            "name": str(self.name),
            "features": list(self.features),
            "labels": list(self.labels),
            "params": dict(self.model_params),
        }

    def compute_stats(self):
        self.proba = self.compute_proba(self.model, self.X)
        self.statistics = self.compute_statistics(
            self.model, self.X, self.Y, self.labels
        )
        self.cv10 = self.compute_10cv(self.model, self.X, self.Y)

    def compute_proba(self, model, X):
        """
        Compute proba + entropy
        """
        proba = model.predict_proba(X)
        proba = pd.DataFrame(proba, columns=model.classes_, index=X.index)
        proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)

        # Calculate label
        proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)

        return proba

    def compute_precision(self, model, X, Y, labels):
        """
        Compute precision score
        """
        f = Y.notna()
        y_pred = model.predict(X[f])
        precision = precision_score(
            list(Y[f]),
            list(y_pred),
            average="micro",
            # pos_label=labels[0]
        )
        return precision

    def compute_statistics(self, model, X, Y, labels):
        """
        Compute statistics simplemodel
        """
        f = Y.notna()
        X = X[f]
        Y = Y[f]
        Y_pred = model.predict(X)
        f1 = f1_score(Y, Y_pred, average=None)
        weighted_f1 = f1_score(Y, Y_pred, average="weighted")
        accuracy = accuracy_score(Y, Y_pred)
        precision = precision_score(
            list(Y[f]),
            list(Y_pred),
            average="micro",
            # pos_label=labels[0]
        )
        macro_f1 = f1_score(Y, Y_pred, average="macro")
        statistics = {
            "f1": list(f1),
            "weighted_f1": weighted_f1,
            "macro_f1": macro_f1,
            "accuracy": accuracy,
            "precision": precision,
        }
        return statistics

    def compute_10cv(self, model, X, Y):
        """
        Compute 10-CV for simplemodel
        TODO : check if ok
        """
        f = Y.notna()
        X = X[f]
        Y = Y[f]
        num_folds = 10
        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
        predicted_labels = cross_val_predict(model, X, Y, cv=kf)
        Y_pred = cross_val_predict(model, X, Y, cv=kf)
        weighted_f1 = f1_score(Y, Y_pred, average="weighted")
        accuracy = accuracy_score(Y, Y_pred)
        macro_f1 = f1_score(Y, Y_pred, average="macro")
        r = {
            "weighted_f1": round(weighted_f1, 3),
            "macro_f1": round(macro_f1, 3),
            "accuracy": round(accuracy, 3),
        }
        return r
