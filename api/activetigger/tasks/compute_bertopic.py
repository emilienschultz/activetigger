import datetime
import json
import shutil
from pathlib import Path
from string import punctuation

import hdbscan  # type: ignore[import]
import pandas as pd
import stopwordsiso as stopwords  # type: ignore[import]
from bertopic import BERTopic  # type: ignore[import]
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore[import]
from slugify import slugify
from simplemma import lemmatize

from activetigger.datamodels import BertopicParamsModel
from activetigger.tasks.base_task import BaseTask
from activetigger.tasks.compute_sbert import ComputeSbert

# accelerate UMAP
try:
    import cuml  # type: ignore[import-not-found]

    CUMl_AVAILABLE = True
except ImportError:
    print("CuML not installed")
    CUMl_AVAILABLE = False
import umap  # type: ignore[import]

"""
Rational : 
- Get texts / existing embeddings / parameters
- Compute embeddings if not provided
- Compute the projection in 2D
- Compute the BERTOPIC pipeline
- Save the results to display
"""

# TODO : multicolumns for text
# TODO : manage special case of embeddings of trainset
# TODO : add the language specific stopwords removal

class CustomLemmatizer:
    """An object to apply the lemmatize function. 
    """
    
    def __init__(self, language, stop_words)->None:
        self.__language = language,
        self.__stop_words = stop_words
    
    def __call__(self, doc : str)->list[str]:
        """
        """
        doc = doc.lower()
        doc = "".join([c for c in doc if c not in punctuation])
        out = []
        for word in doc.split(" "):
            if (word not in self.__stop_words) and (len(word) > 0):
                try:
                    lemma = lemmatize(word, lang = self.__language)
                except Exception as e:
                    # if lemmatization failed, skip it and use the word instead
                    lemma = word
                    print(f"BERTopic - Lemmatization failed (word: {word}) - Error : {e}")
                if lemma not in self.__stop_words:
                    out += [lemma]
        return out

class ComputeBertopic(BaseTask):
    """
    Compute BERTopic model

    A computation is identitied by its name that should be unique
    and is associated with parameters

    Embeddings are computed if not provided or if force_compute_embeddings is True.
    Embeddings are identified by run name + embedding model name.
    """

    kind = "compute_bertopic"

    def __init__(
        self,
        path_bertopic: Path,
        path_data: Path,
        col_id: str | None,
        col_text: str,
        parameters: BertopicParamsModel,
        name: str | None = None,
        existing_embeddings: Path | None = None,
        cols_embeddings: list[str] | None = None,
        force_compute_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__()
        if not path_data.suffix == ".parquet":
            raise ValueError("File must be a parquet file.")
        if existing_embeddings and not existing_embeddings.suffix == ".parquet":
            raise ValueError("Embeddings file must be a parquet file.")

        # Set parameters
        self.path_data = path_data
        self.col_id = col_id
        self.col_text = col_text
        if name is None:
            name = f"bertopic_{path_data.stem}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.name = slugify(name)
        self.file_name = slugify(path_data.stem)
        self.parameters = parameters
        self.existing_embeddings = existing_embeddings
        self.cols_embeddings = cols_embeddings
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.force_compute_embeddings = force_compute_embeddings

        # Create paths
        self.path_bertopic = path_bertopic
        self.path_bertopic.joinpath("embeddings").mkdir(parents=True, exist_ok=True)

        # Check if the run already exists
        self.path_run = self.path_bertopic.joinpath("runs").joinpath(self.name)
        if self.path_run.exists():
            raise ValueError(f"Run {self.name} already exists, please chose another name.")

    def __call__(self):
        """
        Compute BERTopic model
        """
        try:
            # Initialize the run directory
            self.path_run.mkdir(parents=True, exist_ok=True)
            self.update_progress("Initializing")

            # Load the data from the file
            df = pd.read_parquet(self.path_data)
            if self.col_text not in df.columns:
                raise ValueError(f"Column {self.col_text} not found in the data.")

            # Drop rows with a text length too small
            df = df[df[self.col_text].apply(len) > self.parameters.filter_text_length]
            print(f"Data loaded with {len(df)} rows.")

            # Set the index if col_id is provided
            if self.col_id and self.col_id in df.columns:
                df.set_index(self.col_id, inplace=True)

            # Path for existing embeddings
            path_embeddings = self.path_bertopic.joinpath("embeddings").joinpath(
                f"bertopic_embeddings_{self.file_name}_{slugify(self.parameters.embedding_model)}.parquet"
            )
            # Path for 2D projection
            path_projection = self.path_run.joinpath(
                f"bertopic_projection_{self.file_name}_{slugify(self.parameters.embedding_model)}.parquet"
            )

            # Check if existing embeddings are in the folder
            if not self.existing_embeddings and path_embeddings.exists():
                self.existing_embeddings = path_embeddings

            # interrupt if event is set
            if self.event is not None:
                if self.event.is_set():
                    raise Exception("Process interrupted by user")

            # Compute embeddings if not provided or if forced
            if (not self.existing_embeddings) or self.force_compute_embeddings:
                self.compute_embeddings(df, path_embeddings)
                self.compute_projection(path_embeddings, path_projection)
                self.existing_embeddings = path_embeddings

            # Compute projection if does not exist
            if not path_projection.exists():
                self.compute_projection(self.existing_embeddings, path_projection)

            # Copy the projection to the run directory
            shutil.copy(path_projection, self.path_run.joinpath("projection2D.parquet"))

            # Load embeddings
            self.update_progress("Loading embeddings")
            df_embeddings = pd.read_parquet(self.existing_embeddings)
            if self.cols_embeddings and not all(
                col in df_embeddings.columns for col in self.cols_embeddings
            ):
                raise ValueError("Some embeddings columns not found in the embeddings data.")
            if self.cols_embeddings:
                embeddings = df_embeddings[self.cols_embeddings].values
            else:
                embeddings = df_embeddings.values

            # Test if the embeddings & the texts have the same index
            if not df.index.equals(df_embeddings.index) or len(df) != len(df_embeddings):
                raise ValueError(
                    "The index of the embeddings and the texts do not match. "
                    "Please force the computation of embeddings or check your data."
                )

            # Initialize BERTopic
            self.update_progress("Initializing BERTopic")

            # Dimensionality reduction with UMAP
            try:
                umap_model = cuml.UMAP(
                    n_neighbors=self.parameters.umap_n_neighbors,
                    n_components=self.parameters.umap_n_components,
                    min_dist=self.parameters.umap_min_dist,
                    metric="cosine",
                )
            except Exception as e:
                print(f"CuML UMAP failed: {e}, using standard UMAP instead.")
                umap_model = umap.UMAP(
                    n_neighbors=self.parameters.umap_n_neighbors,
                    n_components=self.parameters.umap_n_components,
                    # min_dist=self.parameters.umap_min_dist, # Removed because 0.0 is the best value to use for clustering - Axel
                    metric="cosine",
                )

            # Clustering with HDBSCAN
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=self.parameters.hdbscan_min_cluster_size,
                metric="euclidean",
                prediction_data=True,
            )

            # Vectorizer to manage stopwords
            try:
                stopwords = self.get_stopwords()
                vectorizer_model = CountVectorizer(
                    tokenizer = CustomLemmatizer(self.parameters.language, stopwords)
                )
            except ValueError:
                vectorizer_model = CountVectorizer()

            topic_model = BERTopic(
                language=self.parameters.language,
                vectorizer_model=vectorizer_model,
                # nr_topics=self.parameters.nr_topics, # Removed because overridden by the hdbscan model - Axel
                # min_topic_size=self.parameters.min_topic_size, # Removed to propose topic reduction later in the pipeline - Axel
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
            )
            
            self.update_progress("Fitting the model")
            # Fit the BERTopic model
            topics, _ = topic_model.fit_transform(
                documents=df[self.col_text],
                embeddings=embeddings,
            )

            # Add outlier reduction
            try:
                if self.parameters.outlier_reduction:
                    print("Reducing outliers")
                    topics = topic_model.reduce_outliers(df[self.col_text], topics)
            except Exception as e:
                print(f"Error during outlier reduction: {e}")

            # interrupt if event is set
            if self.event is not None:
                if self.event.is_set():
                    raise Exception("Process interrupted by user")

            # Add the topics to the DataFrame
            df["cluster"] = topics

            # Save the topics and documents informations
            topics_df : pd.DataFrame = topic_model.get_topic_info()
            if self.parameters.outlier_reduction : 
                topics_df = topics_df.loc[topics_df.Topic != -1, :]
            topics_df.to_csv(self.path_run.joinpath("bertopic_topics.csv"))
            df["cluster"].to_csv(self.path_run.joinpath("bertopic_clusters.csv"))
            parameters = {
                "bertopic_params": self.parameters.model_dump(),
                "col_text": self.col_text,
                "col_id": self.col_id,
                "name": self.name,
                "timestamp": self.timestamp,
                "path_data": str(self.path_data),
                "path_embeddings": str(self.existing_embeddings),
                "path_projection": str(path_projection),
            }
            with open(self.path_run.joinpath("params.json"), "w") as f:
                json.dump(parameters, f)

            self.path_run.joinpath("progress").unlink(missing_ok=True)
            return None
        except Exception as e:
            # Case an error happens
            if self.path_run.exists():
                shutil.rmtree(self.path_run)
            raise e

    def update_progress(self, message: str) -> None:
        """
        Update the progress of the task
        """
        with open(self.path_run.joinpath("progress"), "w") as f:
            f.write(message)

    def get_stopwords(self) -> list[str]:
        """
        Get the stopwords for the specified language
        """
        if self.parameters.language not in stopwords.langs():
            raise ValueError(f"Unsupported language {self.parameters.language}")
        return list(stopwords.stopwords(self.parameters.language))

    def compute_embeddings(self, df: pd.DataFrame, path_embeddings: Path) -> None:
        """
        Compute the embeddings using the SBERT model
        """
        if self.parameters.embedding_kind != "sentence_transformers":
            raise ValueError("Only sentence_transformers embeddings are supported for BERTopic.")
        if path_embeddings.exists():
            path_embeddings.unlink()
        self.update_progress(f"Computing embeddings with {self.parameters.embedding_model}")
        embeddings = ComputeSbert(
            texts=df[self.col_text],
            path_process=self.path_bertopic,
            model=self.parameters.embedding_model,
            batch_size=32,
            min_gpu=1,
            path_progress=self.path_run.joinpath("progress"),
        )
        # transmit the event to allow interruption
        embeddings.event = self.event
        # launch computation
        computed = embeddings()
        # save the embeddings to a file
        computed.to_parquet(path_embeddings)

    def compute_projection(self, path_embeddings: Path, path_projection: Path):
        """
        Reduce the dimensionality of the embeddings if needed.
        """
        try:
            reducer = cuml.UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine")
            print("Using cuML for UMAP computation")
        except Exception:
            reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.1, metric="cosine")
            print("Using standard UMAP for computation")
        embeddings = pd.read_parquet(path_embeddings)
        reduced_embeddings = reducer.fit_transform(embeddings)
        df_reduced = pd.DataFrame(reduced_embeddings, index=embeddings.index, columns=["x", "y"])
        df_reduced.to_parquet(path_projection)
