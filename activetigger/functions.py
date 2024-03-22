import pandas as pd
from pandas import DataFrame, Series
import fasttext
import spacy
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.manifold import TSNE
import umap
import bcrypt


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
                 params:dict):
    """
    Compute UMAP
    """
    scaled_features = StandardScaler().fit_transform(features)
    reducer = umap.UMAP(**params)
    reduced_features = reducer.fit_transform(scaled_features)
    df = pd.DataFrame(reduced_features, index = features.index)
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
