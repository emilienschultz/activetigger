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
from sklearn.metrics import precision_score
from datamodels import BertModelModel

class BertModel():
    """
    Managing BertModel

    TODO : metrics
    TODO : tests
    TODO : logs

    """

    def __init__(self, path:Path) -> None:
        """
        All the data are sorted in path/bert/$NAME
        """
        self.name:str|None = None
        self.path:Path = Path(path) / "bert"
        if not self.path.exists():
            os.mkdir(self.path)
        self.model_name: None|str = None
        self.tokenizer: None|AutoTokenizer = None
        self.model: None|AutoModelForSequenceClassification = None
        self.pred: None|DataFrame = None
        self.params: None|dict = None
        self.params_default = self.params = {
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
        self.available = ["microsoft/Multilingual-MiniLM-L12-H384",
                          "almanach/camembert-base"]
        # temporary (all available models)
        self.trained:list = os.listdir(self.path)

    def start_training(self, content:BertModelModel):
        """
        Manage initiating request
        """
        return {"error":"Pas encore implémenté"}

    def train_bert(self,
               name:str,
               df:DataFrame,
               col_text:str,
               col_label:str,
               model_name:str = "microsoft/Multilingual-MiniLM-L12-H384",
               params:dict = {},
               test_size:float = 0.2):
        """
    Train a bert modem
    Parameters:
    ----------
    name (str): name of the model
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    model_name (str): model to use
    params (dict) : training parameters
    """
        self.name = name
        self.model_name = model_name
        
        if len(params) == 0:
            self.params = self.params_default
            params = self.params

        current_path = self.path / self.name
        if not current_path.exists():
            os.mkdir(current_path)

        logging.basicConfig(filename='predict.log',
                            format='%(asctime)s %(message)s',
                            encoding='utf-8', level=logging.DEBUG)
        logging.info(f"Start training {self.model_name}")

        labels = sorted(list(df[col_label].unique())) # alphabetical order
        label2id = {j:i for i,j in enumerate(labels)}
        id2label = {i:j for i,j in enumerate(labels)}
        df["labels"] = df[col_label].copy().replace(label2id)
        df["text"] = df[col_text]
        df = datasets.Dataset.from_pandas(df[["text", "labels"]])

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Tokenize
        print()
        if params["adapt"]:
            df = df.map(lambda e: self.tokenizer(e['text'], truncation=True, padding=True, max_length=512), batched=True)
        else:
            df = df.map(lambda e: self.tokenizer(e['text'], truncation=True, padding="max_length", max_length=512), batched=True)

        # Build test dataset
        df = df.train_test_split(test_size=test_size) #stratify_by_column="label"
        logging.info(f"Train/test dataset created")

        # Model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                                num_labels = len(labels),
                                                                id2label = id2label,
                                                                label2id = label2id)
        if (params["gpu"]):
            self.model.cuda()

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
        
        logging.info(f"Start training")
        trainer = Trainer(model=self.model, 
                         args=training_args, 
                         train_dataset=df["train"], 
                         eval_dataset=df["test"])
        trainer.train()

        # save model
        self.model.save_pretrained(current_path)
        self.trained.append(self.name)
        logging.info(f"Model saved {current_path}")

        # remove intermediate steps
        shutil.rmtree(current_path / "train")

        return True

    def load(self, name:str):
        """
        Load already trained model
        """
        if not (self.path / name / "config.json").exists():
            raise FileNotFoundError("model not defined")

        with open(self.path / name / "config.json", "r") as jsonfile:
            modeltype = json.load(jsonfile)["_name_or_path"]

        self.model_name = modeltype
        self.tokenizer = AutoTokenizer.from_pretrained(modeltype)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path / name)

        return True

    def predict(self, df, col_text = "text", gpu=False, batch = 128):
        """
        Predictor from a model
        Parameters
        ----------
        df (DataFrame): data
        col_text (str): column of the textual data
        gpu (bool): optional, enable GPU
        """

        logging.basicConfig(filename='predict.log',
                            format='%(asctime)s %(message)s',
                            encoding='utf-8', level=logging.DEBUG)
        
        logging.info("Load model")

        if self.model is None : 
            self.load()

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
                inputs = inputs.to("cuda")
            with torch.no_grad():
                outputs = self.model(**chunk)
            res = outputs[0]
            if gpu:
                res = res.cpu()
            res = res.softmax(1).detach().numpy()
            predictions.append(res)
            logging.info(f"{round(100*len(res)/len(df))}% predicted")

        # Tidy data
        pred = pd.DataFrame(np.concatenate(predictions), 
                            columns=self.model.config.labels.keys())
        
        return pred
        # vérifier l'ordre des labels pour éviter les soucis   
    
    def get_params(self):
        """
        Push params
        """
        content = {}
        content["params"] = {"current":self.params,
                             "default":self.params_default}
        content["models"] = {"available":self.available,
                             "current":self.model_name,
                             "trained":self.trained}
        return content

class SimpleModel():
    """
    Managing simple models
    (params/fit/predict)

    Comment : the simplemodel can be empty
    to still access to parameters
    """
    def __init__(self,
                 model: str|None=None,
                 data: DataFrame|None = None,
                 col_label: str|None = None,
                 col_predictors: list|None = None,
                 standardize: bool = False,
                 model_params: dict|None = None
                 ):
        """
        Initialize simpe model for data
        model (str): type of models
        data (DataFrame): dataset
        col_label (str): column of the tags
        predictor (list): columns of predictors
        standardize (bool): predictor standardisation
        model_params: parameters for models
        """

        # logique : des modèles avec des valeurs par défauts
        # ou bien initialisés avec une chaine d'options
        # TODO: vérifier quela chaine est bien formée car cela a des conséquences

        self.available_models = {
            #"simplebayes": {
            #        "distribution":"multinomial",
            #        "smooth":1,
            #        "prior":"uniform"
            #    },
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
        
        self.name = model
        self.df = None
        self.col_label = None
        self.col_predictors = None
        self.X = None
        self.Y = None
        self.labels = None
        self.model = None
        self.proba = None
        self.precision = None
        self.normalize = None
        self.model_params = None

        # Initialize data
        if data is not None and col_predictors is not None:
            self.load_data(data, col_label, col_predictors, standardize)

        # Initialize model
        if self.name in self.available_models:
            if model_params is None:
                self.model_params = self.available_models[self.name]
            else:
                self.model_params = model_params
        
        # Fit model if everything available
        if (not self.X is None) & (not self.Y is None) & (not self.name is None):
            self.fit_model()

    def __repr__(self) -> str:
        return str(self.name)

    def load_data(self, 
                  data, 
                  col_label, 
                  col_predictors,
                  standardize):
        # Build the data set & missing predictors
        # For the moment remove missing predictors
        self.col_label = col_label
        self.col_predictors = col_predictors
        self.normalize = standardize

        f_na = data[self.col_predictors].isna().sum(axis=1)>0      
        if f_na.sum()>0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        # normalize data
        if standardize:
            df_pred = self.standardize(data[~f_na][self.col_predictors])
        else:
            df_pred = data[~f_na][self.col_predictors]

        # create global dataframe
        self.df = pd.concat([data[~f_na][self.col_label],df_pred],axis=1)
    
        # data for training
        f_label = self.df[self.col_label].notnull()
        self.Y = self.df[f_label][self.col_label]
        self.X = self.df[f_label][self.col_predictors]
        self.labels = self.Y.unique()

    def fit_model(self):
        """
        Fit model

        TODO: add naive bayes
        """

        # Select model
        if self.name == "knn":
            self.model = KNeighborsClassifier(n_neighbors=self.model_params["n_neighbors"])

        if self.name == "lasso":
            self.model = LogisticRegression(penalty="l1",
                                            solver="liblinear",
                                            C = self.model_params["C"])
        """
        if self.name == "naivebayes":
            if not "distribution" in self.model_params:
                raise TypeError("Missing distribution parameter for naivebayes")
            
        # only dfm as predictor
            alpha = 1
            if "smooth" in self.model_params:
                alpha = self.model_params["smooth"]
            fit_prior = True
            class_prior = None
            if "prior" in self.model_params:
                if self.model_params["prior"] == "uniform":
                    fit_prior = True
                    class_prior = None
                if self.model_params["prior"] == "docfreq":
                    fit_prior = False
                    class_prior = None #TODO
                if self.model_params["prior"] == "termfreq":
                    fit_prior = False
                    class_prior = None #TODO
 
            if self.model_params["distribution"] == "multinomial":
                self.model = MultinomialNB(alpha=alpha,
                                           fit_prior=fit_prior,
                                           class_prior=class_prior)
            elif self.model_params["distribution"] == "bernouilli":
                self.model = BernoulliNB(alpha=alpha,
                                           fit_prior=fit_prior,
                                           class_prior=class_prior)
            self.model_params = {
                                    "distribution":self.model_params["distribution"],
                                    "smooth":alpha,
                                    "prior":"uniform"
                                }
        """

        if self.name == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            self.model = LogisticRegression(penalty='l2', 
                                            solver='lbfgs',
                                            C = self.model_params["cost"])

        if self.name == "randomforest":
            # params  Num. trees mtry  Sample fraction
            #Number of variables randomly sampled as candidates at each split: 
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            self.model = RandomForestClassifier(n_estimators=self.model_params["n_estimators"], 
                                                random_state=42,
                                                max_features=self.model_params["max_features"])

        # Fit modelmax_features
        self.model.fit(self.X, self.Y)
        self.proba = pd.DataFrame(self.predict_proba(self.df[self.col_predictors]),
                                 columns = self.model.classes_)
        self.precision = self.compute_precision()

    def update(self,content):
        """
        Update the model
        """
        self.name = content["current"]
        self.model_params = content["parameters"]
        self.fit_model()
        return True

    def standardize(self,df):
        """
        Apply standardization
        """
        scaler = StandardScaler()
        df_stand = scaler.fit_transform(df)
        return pd.DataFrame(df_stand,columns=df.columns,index=df.index)

    def compute_precision(self):
        """
        Compute precision score
        """
        y_pred = self.model.predict(self.X)
        precision = precision_score(list(self.Y), list(y_pred),pos_label=self.labels[0])
        return 
    
    def predict_proba(self,X):
        proba = self.model.predict_proba(X)
        return proba
    
    def get_predict(self,rows:str="all"):
        """
        Return predicted proba
        """
        if rows == "tagged":
            return self.proba[self.df[self.col_label].notnull()]
        if rows == "untagged":
            return self.proba[self.df[self.col_label].isnull()]
        
        return self.proba
    
    def get_params(self):
        params = {
            "available":self.available_models,
            "current":self.name,
            "predictors":self.col_predictors,
            "parameters":self.model_params
            # add params e.g. standardization
        }
        return params

        
