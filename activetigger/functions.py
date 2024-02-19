import pandas as pd
from pandas import DataFrame, Series
import fasttext
import spacy
import logging
import json
from pathlib import Path
import torch
import numpy as np
import shutil
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments, TrainerCallback
import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score

# nexts steps : 
# - deal multiple schemes

class BertModel():
    """
    Managing BertModel
    """

    def __init__(self,path:str) -> None:
        self.path:Path = Path(path) / "bert"
        self.model_name: None|str = None
        self.tokenizer: None|AutoTokenizer = None
        self.model: None|AutoModelForSequenceClassification = None
        self.pred: None|DataFrame = None

    def train_bert(self, df:DataFrame,
               col_text:str,
               col_label:str,
               model_name:str = "microsoft/Multilingual-MiniLM-L12-H384",
               params:dict = {},
               test_size:float = 0.2):
        """
    Train a bert modem
    Parameters:
    ----------
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    model_name (str): model to use
    params (dict) : training parameters
    """
        self.model_name = model_name
        
        if len(params) == 0:
            params = {
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

        labels = sorted(list(df[col_label].unique())) # alphabetical order
        label2id = {j:i for i,j in enumerate(labels)}
        id2label = {i:j for i,j in enumerate(labels)}
        df["labels"] = df[col_label].copy().replace(label2id)
        df["text"] = df[col_text]
        df = datasets.Dataset.from_pandas(df[["text", "labels"]])

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Tokenize
        if params["adapt"]:
            df = df.map(lambda e: self.tokenizer(e['text'], truncation=True, padding=True, max_length=512), batched=True)
        else:
            df = df.map(lambda e: self.tokenizer(e['text'], truncation=True, padding="max_length", max_length=512), batched=True)

        # Build test dataset
        df = df.train_test_split(test_size=test_size) #stratify_by_column="label"

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
            output_dir = self.path / "train",
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
        
        trainer = Trainer(model=self.model, 
                         args=training_args, 
                         train_dataset=df["train"], 
                         eval_dataset=df["test"])

        trainer.train()

        # save model
        self.model.config.labels = label2id
        self.model.save_pretrained(self.path)

        # remove intermediate steps
        shutil.rmtree(self.path / "train")

        return True

    def load(self):
        """
        Load already trained model
        """
        if self.path is None:
            raise FileNotFoundError("path not defined")
        if not (self.path / "config.json").exists():
            raise FileNotFoundError("model not definsed")

        with open(self.path / "config.json", "r") as jsonfile:
            modeltype = json.load(jsonfile)["_name_or_path"]

        self.tokenizer = AutoTokenizer.from_pretrained(modeltype)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.path)


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

class SimpleModel():
    """
    Managing simple models
    (params/fit/predict)
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
        self.standardize = None
        self.model_params = None

        # Initialize data for the simplemodel
        if data is not None and col_predictors is not None:
            self.load_data(data, col_label, col_predictors, standardize)

        if self.name is not None:
            if model_params is None:
                self.model_params = self.available_models[self.name]
            else:
                self.model_params = model_params

        # Train model on the data
        if self.name in self.available_models:
            self.model_params = model_params
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
        self.standardize = standardize

        f_na = data[self.col_predictors].isna().sum(axis=1)>0        
        if f_na.sum()>0:
            print(f"There is {f_na.sum()} predictor rows with missing values")

        if standardize:
            df_pred = self.standardize(data[~f_na][self.col_predictors])
        else:
            df_pred = data[~f_na][self.col_predictors]

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
                                            C = self.model_params["lasso_params"])
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

        
def to_dfm(texts: Series,
           tfidf:bool=False,
           ngrams:int=1,
           min_term_freq:int=5) -> DataFrame:
    """
    Compute DFM embedding

    TODO : what is "max_doc_freq" + ADD options from quanteda
    https://quanteda.io/reference/dfm_tfidf.html
    """

    if tfidf:
        vectorizer = TfidfVectorizer(ngram_range=(1, ngrams),
                                      min_df=min_term_freq)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, ngrams),
                                      min_df=min_term_freq)

    dtm = vectorizer.fit_transform(texts)
    dtm = pd.DataFrame(dtm.toarray(), 
                       columns=vectorizer.get_feature_names_out())
    return dtm

def tokenize(texts: Series,
             model: str = "fr_core_news_sm")->Series:
    """
    Clean texts with tokenization to facilitate word count
    """

    nlp = spacy.load(model)
    def tokenize(t,nlp):
        return " ".join([str(token) for token in nlp.tokenizer(t)])
    textes_tk = texts.apply(lambda x : tokenize(x,nlp))
    return textes_tk

def to_fasttext(texts: Series,
                 model: str = "/home/emilien/models/cc.fr.300.bin") -> DataFrame:
    """
    Compute fasttext embedding
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    texts_tk = tokenize(texts)
    ft = fasttext.load_model(model)
    emb = [ft.get_sentence_vector(t.replace("\n"," ")) for t in texts_tk]
    df = pd.DataFrame(emb,index=texts.index)
    df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
    return df

def process_fasttext(texts: Series,
                  path:Path,
                  model:str = "/home/emilien/models/cc.fr.300.bin"):
    texts_tk = tokenize(texts)
    ft = fasttext.load_model(model)
    emb = [ft.get_sentence_vector(t.replace("\n"," ")) for t in texts_tk]
    df = pd.DataFrame(emb,index=texts.index)
    df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
    df.to_parquet(path / "fasttext.parquet")


def to_sbert(texts: Series, 
             model:str = "distiluse-base-multilingual-cased-v1") -> DataFrame:
    """
    Compute sbert embedding
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    sbert = SentenceTransformer(model)
    sbert.max_seq_length = 512
    emb = sbert.encode(list(texts))
    emb = pd.DataFrame(emb,index=texts.index)
    emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
    return emb

def process_sbert(texts: Series,
                  path:Path,
                  model:str = "distiluse-base-multilingual-cased-v1"):
    """
    Compute sbert embedding
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    sbert = SentenceTransformer(model)
    sbert.max_seq_length = 512
    emb = sbert.encode(list(texts))
    emb = pd.DataFrame(emb,index=texts.index)
    emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
    emb.to_parquet(path / "sbert.parquet")


### TO REMOVE EVENTALLY

def train_bert(df:DataFrame,
               col_text:str,
               col_label:str,
               path:str,
               model_name:str = "microsoft/Multilingual-MiniLM-L12-H384",
               params:dict|None = None,
               test_df:float|DataFrame = 0.2):
    """
    Train a bert modem
    Parameters:
    ----------
    df (DataFrame): labelled data
    col_text (str): text column
    col_label (str): label column
    path (str): repertory to save
    model_name (str): model to use
    params (dict) : training parameters
    test_df (float or DataFrame): test of the model 
    """
    
    if params is None:
        params = {
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
    
    path = Path(path) / "bert"
    
    labels = sorted(list(df[col_label].unique())) # alphabetical order
    labels2id = {j:i for i,j in enumerate(labels)}
    df["label"] = df[col_label].replace(labels2id)
    df["text"] = df[col_text]
    df = datasets.Dataset.from_pandas(df[["text", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tokenize
    if params["adapt"]:
        df = df.map(lambda e: tokenizer(e['text'], truncation=True, padding=True, max_length=512), batched=True)
    else:
        df = df.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length", max_length=512), batched=True)

    # Build test dataset
    if type(test_df) is float:
        df = df.train_test_split(test_size=test_df) #stratify_by_column="label"
    else:
        test_df["label"] = test_df[col_label].replace(labels2id)
        test_df["text"] = test_df[col_text]
        test_df = datasets.Dataset.from_pandas(test_df[["text", "label"]])
        datasets.combine.DatasetDict({"train":df,"test":test_df})

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, 
                                                               num_labels = len(labels))
    if (params["gpu"]):
      model.cuda()

    total_steps = (params["epochs"] * len(df["train"])) // (params["batchsize"] * params["gradacc"])
    warmup_steps = (total_steps) // 10
    eval_steps = total_steps // params["eval"]
    
    training_args = TrainingArguments(
        output_dir = path / "train",
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
    
    trainer = Trainer(model=model, 
                      args=training_args, 
                      train_dataset=df["train"], 
                      eval_dataset=df["test"])

    """
    the_session = model_name

    # GO TO A SUBPROCESS MANAGEMENT
    class HaltCallback(TrainerCallback):
      "A callback that checks for _stop file to interrupt training"
      
      def on_step_begin(self, args, state, control, **kwargs):
          if os.path.exists(the_session + "_stop"):
              #print("\nHalted by user.\n")
              control.should_training_stop = True
              return(control)
          else:
              #print("\nNot halted by user.\n")

    trainer.add_callback(HaltCallback)
    """
    trainer.train()

    # save model
    model.config.labels = labels2id
    model.save_pretrained(path)

    # remove intermediate steps
    shutil.rmtree(path / "train")

    # make predictions

    return model

def predict(path,df,col_text = "text", gpu=False, batch = 128):
    """
    Predictor from a model
    Parameters
    ----------
    path (str): path to the model
    df (DataFrame): data
    col_text (str): column of the textual data
    gpu (bool): optional, enable GPU
    """

    logging.basicConfig(filename='predict.log',
                        format='%(asctime)s %(message)s',
                        encoding='utf-8', level=logging.DEBUG)
    
    logging.info("Load model")
    # Load model
    with open(Path(path) / "config.json", "r") as jsonfile:
        modeltype = json.load(jsonfile)["_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(modeltype)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    if gpu:
        model.cuda()

    # Start prediction with batches
    predictions = []
    logging.info(f"Start prediction with {len(df)} entries")
    for chunk in [df[col_text][i:i+batch] for i in range(0,df.shape[0],batch)]:
        chunk = tokenizer(list(chunk), 
                          padding=True, 
                          truncation=True, 
                          max_length=512, 
                          return_tensors="pt")
        if gpu:
            inputs = inputs.to("cuda")
        with torch.no_grad():
            outputs = model(**chunk)
        res = outputs[0]
        if gpu:
            res = res.cpu()
        res = res.softmax(1).detach().numpy()
        predictions.append(res)
        logging.info(f"{round(100*len(res)/len(df))}% predicted")

    # Tidy data
    pred = pd.DataFrame(np.concatenate(predictions), 
                        columns=model.config.labels.keys())
    
    return pred
    # vérifier l'ordre des labels pour éviter les soucis    

def log_process(name, log_path):
    """
    start a log for a process
    """
    logger = logging.getLogger(name)
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger