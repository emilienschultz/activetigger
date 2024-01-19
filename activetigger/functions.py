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

# nexts steps : 
# - deal multiple schemes

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

