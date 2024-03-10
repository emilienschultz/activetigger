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
import umap


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


def compute_umap(features:DataFrame, 
                 params:dict, 
                 path:Path, 
                 name:str):
    """
    Compute UMAP
    """
    scaled_features = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(**params)
    reduced_features = reducer.fit_transform(scaled_features)
    df = pd.DataFrame(reduced_features, index = features.index)
    df.to_parquet(path / f"{name}.parquet")
    return None

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