import pandas as pd
from pandas import DataFrame, Series
import logging
import json
import os
from pathlib import Path
import torch
import numpy as np
import shutil
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from multiprocessing import Process
from datetime import datetime

logging.basicConfig(filename = "log",
                            format='%(asctime)s %(message)s',
                            encoding='utf-8', level=logging.DEBUG)


class BertModel():
    """
    Manage one bertmodel
    """

    def __init__(self, 
                 name:str, 
                 path:Path, 
                 base_model:str|None = None, 
                 params:dict = {}) -> None:
        self.name:str = name
        self.path:Path = path
        self.base_model:str|None = base_model
        self.tokenizer = None
        self.model = None
        self.params:dict = params
        self.status:str = "initializing"
        self.pred:DataFrame|None = None
        self.timestamp:datetime = datetime.now()

    def __repr__(self) -> str:
        return f"{self.name} - {self.base_model}"

    def load(self):
        """
        Load trained model from files
        """
        if not (self.path / "config.json").exists():
            raise FileNotFoundError("model not defined")

        with open(self.path / "config.json", "r") as jsonfile:
            modeltype = json.load(jsonfile)["_name_or_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(modeltype)
        self.model =  AutoModelForSequenceClassification.from_pretrained(self.path)
        self.status = "loaded"

        # Load prediction if available
        if (self.path / "predict.csv").exists():
            self.pred = pd.read_csv(self.path / "predict.csv")

    def predict(self, 
                df:DataFrame, 
                col_text:str, 
                gpu:bool = False, 
                batch:int = 128):
        """
        Predict from a model
        + probabilities
        + entropy
        """

        if (self.model is None) or (self.tokenizer is None):
            self.load()

        if (self.model is None) or (self.tokenizer is None):
            return {"error":"Model not loaded"}

        if gpu:
            self.model.cuda()

        # Start prediction with batches
        predictions = []
        logging.info(f"Start prediction with {len(df)} entries")
        for chunk in [df[col_text][i:i+batch] for i in range(0,df.shape[0],batch)]:
            chunk = self.tokenizer(list(chunk), 
                            padding=True, 
                            truncation=True, 
                            max_length=512, 
                            return_tensors="pt")
            if gpu:
                chunk = chunk.to("cuda")
            with torch.no_grad():
                outputs = self.model(**chunk)
            res = outputs[0]
            if gpu:
                res = res.cpu()
            res = res.softmax(1).detach().numpy()
            predictions.append(res)
            logging.info(f"{round(100*len(res)/len(df))}% predicted")

        # To DataFrame
        pred = pd.DataFrame(np.concatenate(predictions), 
                            columns=sorted(list(self.model.config.label2id.keys())))

        # Calculate entropy
        entropy = -1 * (pred * np.log(pred)).sum(axis=1)
        pred["entropy"] = entropy

        # Calculate label
        pred["prediction"] = pred.drop(columns="entropy").idxmax(axis=1)

        # write the file
        pred.to_csv(self.path / "predict.csv")

        return pred

class BertModels():
    """
    Managing bertmodel training

    Comments:
        All the data are sorted in path/bert/$NAME

    TODO : std.err in the logs for processes
    """

    def __init__(self, path:Path) -> None:
        self.params_default = {
                        "batchsize":4,
                        "gradacc":1,
                        "epochs":3,
                        "lrate":5e-05,
                        "wdecay":0.01,
                        "best":True,
                        "eval":10,
                        "gpu":False,
                        "adapt":True
                    }
        self.base_models = [
                        "microsoft/Multilingual-MiniLM-L12-H384",
                        "almanach/camembert-base"
                        ]
        
        self.path:Path = Path(path) / "bert"
        if not self.path.exists():
            os.mkdir(self.path)

        # keep current processes (one by user max)
        self.processes:dict = {}

    def __repr__(self) -> str:
        return f"Trained models : {self.trained()}"

    def trained(self) -> dict:
        """
        Trained bert
        + if prediction available
        + compression if available / launch it
        """
        r:dict = {}
        if self.path.exists(): #if bert models have been trained
            all_files = os.listdir(self.path)
            trained = [i for i in all_files if os.path.isdir(self.path / i) and (self.path / i / "finished").exists()]
            #trained = [i for i in trained if i[0]!="_"] #skip temporary training
            for i in trained:
                predict = False
                compressed = False
                # test if prediction available
                if (self.path / i / "predict.csv").exists(): 
                    predict = True
                # test if compression available
                if (self.path / f"{i}.tar.gz").exists():
                    compressed = True
                else :
                    self.start_compression(i)
                scheme = i.split("__")[-1] #scheme after __
                if not scheme in r: 
                    r[scheme] = []
                r[scheme].append((i,predict, compressed))
        return r
    
    def training(self) -> dict:
        """
        Currently under training
        """
        #return {u:self.processes[u][0].status for u in self.processes}
        return {u:self.processes[u][0].name for u in self.processes if self.processes[u][0].status == "training"}

    def start_training_process(self,
               name:str,
               user:str,
               scheme:str,
               df:DataFrame,
               col_text:str,
               col_label:str,
               base_model:str|None = None,
               params:dict|None = None,
               test_size:float|None = None) -> dict:
        """
        Manage the training of a model from the API
        """

        # Check if there is no other competing processes
        # For the moment : 1 active process by user
        if user in self.processes:
            return {"error":"processes already launched, cancel it before"}

        # Set default parameters if needed
        if base_model is None:
            base_model = "almanach/camembert-base"
        if params is None:
            params = self.params_default
        if test_size is None:
            test_size = 0.2

        # name integrating the scheme
        name = f"{name}__{scheme}"

        # Launch as a independant process
        args = {
                "path":self.path,
                "name":name,
                "df":df,
                "col_label":col_label,
                "col_text":col_text,
                "base_model":base_model,
                "params":params,
                "test_size":test_size
                }
        process = Process(target=self.train_bert, 
                          kwargs = args)
        process.start()

        # Update the queue
        b = BertModel(name, self.path / name, base_model)
        b.status = "training"
        self.processes[user] = [b,process]
        return {"success":"bert model on training"}

    def start_predicting_process(self, 
                                 name:str,
                                 user:str, 
                                 df:DataFrame,
                                 col_text:str):
        """
        Start predicting process
        """
        if user in self.processes:
            return {"error":"Processes already launched, cancel it before"}

        if not (self.path / name).exists():
            return {"error":"This model does not exist"}

        b = BertModel(name, self.path / name)

        args = {"df":df, "col_text":col_text}
        process = Process(target=b.predict, 
                          kwargs = args)
        process.start()

        b.status = "predicting"
        self.processes[user] = [b,process]
        return {"success":"bert model predicting"}

    def start_compression(self, name):
        """
        Compress bertmodel as a separate process
        """
        process = Process(target=shutil.make_archive, 
                          args = (self.path / name, 'gztar', self.path))
        process.start()
        print("starting compression")

    def train_bert(self,
               path:Path,
               name:str,
               df:DataFrame,
               col_text:str,
               col_label:str,
               base_model:str,
               params:dict,
               test_size:float) -> bool:
        """
    Train a bert model and write it

    Parameters:
    ----------
    path (Path): path to save the files
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    model (str): model to use
    params (dict) : training parameters
    test_size (dict): train/test distribution
    """

        # pour le moment fichier status.log existe tant que l'entrainement est en cours

        #  create repertory for the specific model
        current_path = path / name
        if not current_path.exists():
            os.mkdir(current_path)

        # logging the process
        log_path = current_path / "status.log"
        logger = logging.getLogger('train_bert_model')
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Start {base_model}")

        # test labels missing values
        if df[col_label].isnull().sum() > 0:
            df = df[df[col_label].notnull()]
            logger.info(f"Missing labels - reducing training data to {len(df)}")

        # test empty texts
        if df[col_text].isnull().sum() > 0:
            df = df[df[col_text].notnull()]
            logger.info(f"Missing texts - reducing training data to {len(df)}")

        print(df)

        # formatting data
        labels = sorted(list(df[col_label].dropna().unique())) # alphabetical order
        label2id = {j:i for i,j in enumerate(labels)}
        id2label = {i:j for i,j in enumerate(labels)}
        df["labels"] = df[col_label].copy().replace(label2id)
        df["text"] = df[col_text]
        df = datasets.Dataset.from_pandas(df[["text", "labels"]])

        tokenizer = AutoTokenizer.from_pretrained(base_model)

        print("tokenize")

        # Tokenize
        if params["adapt"]:
            df = df.map(lambda e: tokenizer(e['text'], truncation=True, padding=True, max_length=512), batched=True)
        else:
            df = df.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length", max_length=512), batched=True)

        # Build test dataset
        df = df.train_test_split(test_size=test_size) #stratify_by_column="label"
        logger.info(f"Train/test dataset created")

        # Model
        bert = AutoModelForSequenceClassification.from_pretrained(base_model, 
                                                                num_labels = len(labels),
                                                                id2label = id2label,
                                                                label2id = label2id)
        
        logger.info(f"Model loaded")

        if (params["gpu"]):
            bert.cuda()

        total_steps = (params["epochs"] * len(df["train"])) // (params["batchsize"] * params["gradacc"])
        warmup_steps = (total_steps) // 10
        eval_steps = total_steps // params["eval"]
        
        training_args = TrainingArguments(
            output_dir = current_path / "train",
            learning_rate=params["lrate"],
            weight_decay=params["wdecay"],
            num_train_epochs=params["epochs"],
            gradient_accumulation_steps=params["gradacc"],
            per_device_train_batch_size=params["batchsize"],
            per_device_eval_batch_size=32,
            warmup_steps=warmup_steps,
            eval_steps=eval_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=eval_steps,
            logging_steps=eval_steps,
            do_eval=True,
            greater_is_better=False,
            load_best_model_at_end=params["best"],
            metric_for_best_model="eval_loss"
            )
        
        logger.info(f"Start training")

        class CustomLoggingCallback(TrainerCallback):
            def on_step_end(self, args, state, control, **kwargs):
                logger.info(f"Step {state.global_step}")

        trainer = Trainer(model=bert, 
                         args=training_args, 
                         train_dataset=df["train"], 
                         eval_dataset=df["test"],
                         callbacks=[CustomLoggingCallback()])
        trainer.train()

        # save model
        bert.save_pretrained(current_path)
        logger.info(f"Model trained {current_path}")

        # remove intermediate steps and logs if succeed
        shutil.rmtree(current_path / "train")
        os.rename(log_path, current_path / "finished")

        return True
    
    def stop_user_training(self,user:str):
        """
        Stop the process of an user
        """
        if not user in self.processes:
            return {"error":"no current processes"}
        self.processes[user][1].terminate() # end process
        shutil.rmtree(self.processes[user][0].path) #delete files
        del self.processes[user] # delete process
        return {"success":"process terminated"}
    
    def rename(self, former_name:str, new_name:str):
        """
        Rename a model (copy it)
        """
        if not (self.path / former_name).exists():
            return {"error":"no model currently trained"}
        if (self.path / new_name).exists():
            return {"error":"this name already exists"}
        if (self.path / former_name / "status.log").exists():
            return {"error":"model not trained completly"}

        # keep the scheme information
        if not "__" in new_name:
            new_name = new_name + "__" + former_name.split("__")[-1]

        os.rename(self.path / former_name, self.path / new_name)

        #shutil.copytree(self.path / former_name, self.path / new_name)
        return {"success":"model renamed"}
            
    def get(self, name:str)-> BertModel|None:
        """
        Get a model (load it if available)
        """
        if not (self.path / name).exists():
            return None
        if (self.path / name / "status.log").exists():
            return None    
        b = BertModel(name, self.path / name)
        b.load()
        return b
    
    def update(self) -> bool:
        """
        Update training queue
        (used in the API)
        # TODO : manage failed processes
        """
        to_del = []
        for u in self.processes:
            b = self.processes[u][0]
            p = self.processes[u][1]
            # test if process completed (training)
            if (b.status == "training") and (not p.is_alive()):
                to_del.append(b.name)
            # test if process completed (predicting)
            if (b.status == "predicting") and (not p.is_alive()):
                to_del.append(b.name)
            print(to_del)
        # Update the current active processes
        self.processes = {u:self.processes[u] for u in self.processes if self.processes[u][0].name not in to_del}
        #self.processes = [b for b in self.processes if b[0].name not in to_del]
        return True
    
    def export_prediction(self, name:str, format:str|None = None):
        """
        Export predict file if exists
        """
        file_name = f"predict.csv"
        if not (self.path / name / file_name).exists():
            return {"error":"file does not exist"}
        return file_name, self.path / name / file_name

    def export_bert(self, name:str):
        """
        Export bert archive if exists
        """
        file_name = f"{name}.tar.gz"
        if not (self.path / file_name).exists():
            return {"error":"file does not exist"}
        return file_name, self.path / file_name
        
 
class SimpleModels():
    """
    Managing simplemodels
    """
    available_models = {
        "liblinear": {
                "cost":1
            },
        "knn" : {
                "n_neighbors":3
            },
        "randomforest": {
                "n_estimators":500,
                "max_features":None
                },
        "lasso": {
                "C":32
                }
            }
    
    def __init__(self):
        self.existing = {}

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
                r[u][s] = {"name":sm.name, 
                           "params":sm.model_params,
                           "features":sm.features,
                           "statistics":sm.statistics
                           }
        return r
        
    def exists(self, user:str, scheme:str):
        """
        Test if a simplemodel exists for a user/scheme
        """
        if user in self.existing:
            if scheme in self.existing[user]:
                return True
        return False

    def get_model(self, user:str, scheme:str):
        """
        Select a specific model in the repo
        """
        if not user in self.existing:
            return "This user has no model"
        if not scheme in self.existing[user]:
            return "The model for this scheme does not exist"
        return self.existing[user][scheme]
    
    def load_data(self, 
                  data, 
                  col_label, 
                  col_predictors,
                  standardize):
        """
        Load data
        """
        f_na = data[col_predictors].isna().sum(axis=1)>0      
        if f_na.sum()>0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize X data
        if standardize:
            df_pred = self.standardize(data[~f_na][col_predictors])
        else:
            df_pred = data[~f_na][col_predictors]

        # create global dataframe with no missing predictor
        df = pd.concat([data[~f_na][col_label],df_pred],axis=1)
    
        # data for training
        #f_label = df[self.col_label].notnull()
        #Y = df[f_label][self.col_label]
        #X = df[f_label][self.col_predictors]
        Y = df[col_label]
        X = df[col_predictors]
        labels = Y.unique()
        
        return X, Y, labels
    
    def fit_model(self, name, X, Y, model_params = None):
        """
        Fit model
        TODO: add naive bayes
        """
        # only keep tagged Y
        f = Y.notnull()
        X = X[f]
        Y = Y[f]
        # default parameters
        if model_params is None:
            model_params = self.available_models[name]

        # Select model
        if name == "knn":
            model = KNeighborsClassifier(n_neighbors=model_params["n_neighbors"])

        if name == "lasso":
            model = LogisticRegression(penalty="l1",
                                            solver="liblinear",
                                            C = model_params["C"])

        if name == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            model = LogisticRegression(penalty='l2', 
                                            solver='lbfgs',
                                            C = model_params["cost"])

        if name == "randomforest":
            # params  Num. trees mtry  Sample fraction
            #Number of variables randomly sampled as candidates at each split: 
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            model = RandomForestClassifier(n_estimators=model_params["n_estimators"], 
                                                random_state=42,
                                                max_features=model_params["max_features"])

        # Fit modelmax_features
        model.fit(X, Y)
        return model, model_params

    def standardize(self,df):
        """
        Apply standardization
        """
        scaler = StandardScaler()
        df_stand = scaler.fit_transform(df)
        return pd.DataFrame(df_stand,columns=df.columns,index=df.index)

    def add_simplemodel(self, 
                        user, 
                        scheme,
                        features,
                        name, 
                        df, 
                        col_labels,
                        col_features,
                        standardize,
                        model_params:dict|None = None):
        """
        A a new simplemodel for a user and a scheme
        """
        X, Y, labels = self.load_data(df, col_labels, col_features, standardize)
        model, model_params = self.fit_model(name, X, Y, model_params)
        sm = SimpleModel(name, X, Y, labels, model, features, standardize, model_params)
        if not user in self.existing:
            self.existing[user] = {}
        self.existing[user][scheme] = sm
        print("model added")

class SimpleModel():
    def __init__(self,
                 name: str,
                 X: DataFrame,
                 Y: DataFrame,
                 labels: list,
                 model,
                 features:list,
                 standardize: bool,
                 model_params: dict|None
                 ) -> None:
        self.name = name
        self.features = features
        self.X = X
        self.Y = Y
        self.labels = labels
        self.model = model
        self.model_params = model_params
        self.proba = self.compute_proba(model, X)
        self.standardize = standardize
        self.statistics = self.compute_statistics(model, X, Y, labels)
        self.cv10 = self.compute_10cv(model, X, Y)
        print(self.statistics)

    def json(self):
        """
        Return json representation
        """
        return {
            "name":str(self.name),
            "features":list(self.features),
            "labels":list(self.labels),
            "params":dict(self.model_params)
        }

    def compute_proba(self, model, X):
        """
        Compute proba + entropy
        """
        proba = model.predict_proba(X)
        proba = pd.DataFrame(proba, 
                             columns = model.classes_,
                             index=X.index)
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
        precision = precision_score(list(Y[f]), 
                                    list(y_pred),
                                    pos_label=labels[0])
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
        weighted_f1 = f1_score(Y, Y_pred, average='weighted')
        accuracy = accuracy_score(Y, Y_pred)
        precision = precision_score(list(Y[f]), 
                                    list(Y_pred),
                                    pos_label=labels[0])
        macro_f1 = f1_score(Y, Y_pred, average='macro')
        statistics = {
                    "f1":list(f1),
                    "weighted_f1":weighted_f1,
                    "macro_f1":macro_f1,
                    "accuracy":accuracy,
                    "precision":precision
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
        weighted_f1 = f1_score(Y, Y_pred,average = "weighted")
        accuracy = accuracy_score(Y, Y_pred)
        macro_f1 = f1_score(Y, Y_pred,average = "macro")
        r = {"weighted_f1":round(weighted_f1,3), 
             "macro_f1":round(macro_f1,3),
             "accuracy":round(accuracy,3)}
        return r

#         self.available_models = {
#                 #"simplebayes": {
#                 #        "distribution":"multinomial",
#                 #        "smooth":1,
#                 #        "prior":"uniform"
#                 #    },            C = self.model_params["C"])
#         if self.name == "naivebayes":
#             if not "distribution" in self.model_params:
#                 raise TypeError("Missing distribution parameter for naivebayes")
            
#         # only dfm as predictor
#             alpha = 1
#             if "smooth" in self.model_params:
#                 alpha = self.model_params["smooth"]
#             fit_prior = True
#             class_prior = None
#             if "prior" in self.model_params:
#                 if self.model_params["prior"] == "uniform":
#                     fit_prior = True
#                     class_prior = None
#                 if self.model_params["prior"] == "docfreq":
#                     fit_prior = False
#                     class_prior = None #TODO
#                 if self.model_params["prior"] == "termfreq":
#                     fit_prior = False
#                     class_prior = None #TODO
 
#             if self.model_params["distribution"] == "multinomial":
#                 self.model = MultinomialNB(alpha=alpha,
#                                            fit_prior=fit_prior,
#                                            class_prior=class_prior)
#             elif self.model_params["distribution"] == "bernouilli":
#                 self.model = BernoulliNB(alpha=alpha,
#                                            fit_prior=fit_prior,
#                                            class_prior=class_prior)
#             self.model_params = {
#                                     "distribution":self.model_params["distribution"],
#                                     "smooth":alpha,
#                                     "prior":"uniform"
#                                 }