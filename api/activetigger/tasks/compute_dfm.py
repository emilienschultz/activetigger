import pandas as pd
import spacy
from pandas import DataFrame, Series
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from activetigger.tasks.base_task import BaseTask


class ComputeDfm(BaseTask):
    """
    Compute sbert feature
    """

    kind = "compute_feature_sbert"

    def __init__(
        self,
        texts: Series,
        tfidf: bool = False,
        ngrams: int = 1,
        min_term_freq: int = 5,
        max_term_freq: int | float = 1.0,
        log: bool = False,
        language: str = "en",
        norm=None,
        **kwargs,
    ):
        self.texts = texts
        self.tfidf = tfidf
        self.ngrams = ngrams
        self.min_term_freq = min_term_freq
        self.max_term_freq = max_term_freq
        self.log = log
        self.language = language
        self.norm = norm

    def __call__(self) -> DataFrame:
        """
        Compute Document Term Matrix

        Norm :  None, l1, l2
        sublinear_tf : log
        Pas pris en compte : DFM : Min Docfreq
        https://quanteda.io/reference/dfm_tfidf.html
        """

        # load stopwords
        if self.language == "fr":
            nlp = spacy.blank("en")
            stop_words = list(nlp.Defaults.stop_words)
        else:
            nlp = spacy.blank("en")
            stop_words = list(nlp.Defaults.stop_words)

        # compute matrix
        if self.tfidf:
            vectorizer = TfidfVectorizer(
                ngram_range=(1, self.ngrams),
                min_df=self.min_term_freq,
                sublinear_tf=self.log,
                norm=self.norm,
                max_df=self.max_term_freq,
                stop_words=stop_words,
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=(1, self.ngrams),
                min_df=self.min_term_freq,
                max_df=self.max_term_freq,
                stop_words=stop_words,
            )

        dtm = vectorizer.fit_transform(self.texts)
        names = vectorizer.get_feature_names_out()
        dtm = pd.DataFrame(dtm.toarray(), columns=names, index=self.texts.index)
        return {"success": dtm}
