import pandas as pd
from pandas import DataFrame, Series
import fasttext
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score, KFold, cross_val_predict
from sklearn.manifold import TSNE
import umap
import bcrypt
import numpy as np

def get_hash(text:str):
    """
    Hash string    
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(text.encode(), salt)
    return hashed

def compare_to_hash(text:str, hash:str|bytes):
    """
    Compare string to hash
    """
    if type(text) is str:
        text = text.encode()
    if type(hash) is str:
        hash = hash.encode()
    r = bcrypt.checkpw(text, hash)
    return r

def process_dfm(texts: Series,
           path:Path,
           tfidf:bool=False,
           ngrams:int=1,
           min_term_freq:int=5,
           max_term_freq:int|float = 1.0,
           log:bool = False,
           norm = None
           ):
    """
    Compute DFM embedding
    
    Norm :  None, l1, l2
    sublinear_tf : log
    Pas pris en compte : DFM : Min Docfreq
    https://quanteda.io/reference/dfm_tfidf.html
    + stop_words
    """
    if tfidf:
        vectorizer = TfidfVectorizer(ngram_range=(1, ngrams),
                                      min_df=min_term_freq,
                                      sublinear_tf = log,
                                      norm = norm,
                                      max_df = max_term_freq)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, ngrams),
                                      min_df=min_term_freq,
                                      max_df = max_term_freq)

    dtm = vectorizer.fit_transform(texts)
    names = vectorizer.get_feature_names_out()
    dtm = pd.DataFrame(dtm.toarray(), 
                       columns = names, 
                       index = texts.index)
    dtm.to_parquet(path / "dfm.parquet")

def tokenize(texts: Series,
             model: str = "fr_core_news_sm")->Series:
    """
    Clean texts with tokenization to facilitate word count
    TODO : faster tokenization ?
    """

    nlp = spacy.load(model)
    def tokenize(t,nlp):
        return " ".join([str(token) for token in nlp.tokenizer(t)])
    textes_tk = texts.apply(lambda x : tokenize(x,nlp))
    return textes_tk

def to_fasttext(texts: Series,
                 model: str) -> DataFrame:
    """
    Compute fasttext embedding
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    texts_tk = tokenize(texts)
    print(model)
    ft = fasttext.load_model(model)
    print("loaded")
    emb = [ft.get_sentence_vector(t.replace("\n"," ")) for t in texts_tk]
    df = pd.DataFrame(emb,index=texts.index)
    df.columns = ["ft%03d" % (x + 1) for x in range(len(df.columns))]
    return df

def process_fasttext(texts: Series,
                  path:Path,
                  model:str):
    texts_tk = tokenize(texts)
    ft = fasttext.load_model(str(model))
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
                 params:dict):
    """
    Compute UMAP
    """
    try:
        scaled_features = StandardScaler().fit_transform(features)
        reducer = umap.UMAP(**params)
        reduced_features = reducer.fit_transform(scaled_features)
        df = pd.DataFrame(reduced_features, index = features.index)
    except:
        df = DataFrame()
    return df

def compute_tsne(features:DataFrame, 
                 params:dict):
    """
    Compute TSNE
    """
    scaled_features = StandardScaler().fit_transform(features)
    reduced_features = TSNE(**params).fit_transform(scaled_features)
    df = pd.DataFrame(reduced_features,index = features.index)
    return df

def fit_model(model, X, Y, labels):
    """
    Fit simplemodel and calculate statistics
    """

    # drop NA values
    f = Y.notnull()
    Xf = X[f]
    Yf = Y[f]

    # fit model
    model.fit(Xf, Yf)

    # compute probabilities
    proba = model.predict_proba(X)
    proba = pd.DataFrame(proba, 
                            columns = model.classes_,
                            index=X.index)
    proba["entropy"] = -1 * (proba * np.log(proba)).sum(axis=1)
    proba["prediction"] = proba.drop(columns="entropy").idxmax(axis=1)


    # compute statistics
    Y_pred = model.predict(Xf)
    f1 = f1_score(Yf, Y_pred, average=None)
    weighted_f1 = f1_score(Yf, Y_pred, average='weighted')
    accuracy = accuracy_score(Yf, Y_pred)
    precision = precision_score(list(Yf), 
                                list(Y_pred),
                                average="micro",
                                )
    macro_f1 = f1_score(Yf, Y_pred, average='macro')
    statistics = {
                "f1":list(f1),
                "weighted_f1":weighted_f1,
                "macro_f1":macro_f1,
                "accuracy":accuracy,
                "precision":precision
                }
    
    # compute 10-crossvalidation
    num_folds = 10
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    Y_pred = cross_val_predict(model, Xf, Yf, cv=kf)
    weighted_f1 = f1_score(Yf, Y_pred,average = "weighted")
    accuracy = accuracy_score(Yf, Y_pred)
    macro_f1 = f1_score(Yf, Y_pred,average = "macro")
    cv10 = {"weighted_f1":round(weighted_f1,3), 
            "macro_f1":round(macro_f1,3),
            "accuracy":round(accuracy,3)}

    r  = {"model":model, 
          "proba":proba, 
          "statistics":statistics, 
          "cv10":cv10
        }
    return r