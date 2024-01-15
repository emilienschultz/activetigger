import pandas as pd
from pandas import DataFrame, Series
import fasttext
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import precision_score



# tokenisation ?


class SimpleModel():
    """
    Managing fit/predict
    """

    def __init__(self,
                 model:str,
                 data:DataFrame,
                 label:str,
                 predictors:list,
                 standardize:bool=False,
                 **kwargs
                 ):
        
        self.available_models = ["simplebayes",
                                 "liblinear",
                                 "knn",
                                 "randomforest",
                                 "lasso"]

        # Data curation
        
        # Drop NaN
        df = data[[label]+predictors].dropna()

        self.Y = df[label]
        self.X = df[predictors]
        self.labels = self.Y.unique()

        if standardize:
            self.standardize()

        # Select model
        if model == "knn":
            n_neighbors = len(self.labels)
            if "n_neighbors" in kwargs:
                n_neighbors = kwargs["n_neighbors"]
            self.model = KNeighborsClassifier(n_neighbors=n_neighbors)

        if model == "lasso":
            # Cfloat, default=1.0
            C = 1
            if "lasso_params" in kwargs:
                C = kwargs["lasso_params"]
            self.model = LogisticRegression(penalty="l1",
                                            solver="liblinear",
                                            C = C)
        if model == "naivebayes":
        # only dfm as predictor
            alpha = 1
            if "smooth" in kwargs:
                alpha = kwargs["smooth"]
            if "prior" in kwargs:
                if kwargs["prior"] == "uniform":
                    fit_prior = True
                    class_prior = None
                if kwargs["prior"] == "docfreq":
                    fit_prior = False
                    class_prior = None #TODO
                if kwargs["prior"] == "termfreq":
                    fit_prior = False
                    class_prior = None #TODO
            if kwargs["distribution"] == "multinomial":
                self.model = MultinomialNB(alpha=alpha,
                                           fit_prior=fit_prior,
                                           class_prior=class_prior)
            elif kwargs["distribution"] == "bernouilli":
                self.model = BernoulliNB(alpha=alpha,
                                           fit_prior=fit_prior,
                                           class_prior=class_prior)

        if model == "liblinear":
            # Liblinear : method = 1 : multimodal logistic regression l2
            C = 32
            if "cost" in kwargs: # params  Liblinear Cost 
                C = kwargs["cost"]
            self.model = LogisticRegression(penalty='l2', 
                                            solver='lbfgs',
                                            C = C)

        if model == "randomforest":
            # params  Num. trees mtry  Sample fraction
            #Number of variables randomly sampled as candidates at each split: 
            # it is “mtry” in R and it is “max_features” Python
            #  The sample.fraction parameter specifies the fraction of observations to be used in each tree
            n_estimators: int = 500
            max_features: None|int = None 
            if "n_estimators" in kwargs: 
                n_estimators = kwargs["n_estimators"]
            if "max_features" in kwargs: 
                max_features = kwargs["max_features"]
            # TODO ADD SAMPLE.FRACTION

            self.model = RandomForestClassifier(n_estimators=100, 
                                                random_state=42,
                                                max_features=max_features)

        # Fit model
        self.model.fit(self.X, self.Y)
        self.precision = self.compute_precision()

    def standardize(self):
        """
        Apply standardization
        """
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def compute_precision(self):
        """
        Compute precision score
        """
        y_pred = self.model.predict(self.X)
        precision = precision_score(list(self.Y), list(y_pred),pos_label=self.labels[0])
        return precision

        
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
    emb = sbert.encode(texts)
    emb = pd.DataFrame(emb,index=texts.index)
    emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
    return emb

