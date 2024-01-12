import pandas as pd
import fasttext
from sentence_transformers import SentenceTransformer

def to_fasttext(texts, model = "/home/emilien/models/cc.fr.300.bin"):
    """
    Compute fasttext embeddings
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """

    ft = fasttext.load_model(model)
    emb = [ft.get_sentence_vector(t.replace("\n"," ")) for t in texts]
    emb = pd.DataFrame(emb,index=texts.index)
    emb.columns = ["ft%03d" % (x + 1) for x in range(len(emb.columns))]
    return emb

def to_sbert(texts, model = "distiluse-base-multilingual-cased-v1"):
    """
    Compute sbert embeddings
    Args:
        texts (pandas.Series): texts
        model (str): model to use
    Returns:
        pandas.DataFrame: embeddings
    """
    sbert = SentenceTransformer(model)
    sbert.max_seq_length = 512
    emb = sbert.encode(texts)
    emb = pd.DataFrame(emb,index=texts.index)
    emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
    return emb
