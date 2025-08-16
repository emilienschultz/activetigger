from pathlib import Path

import pandas as pd
import stopwordsiso as stopwords
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from slugify import slugify

from activetigger.datamodels import BertTopicParamsModel
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


class ComputeBertTopic(BaseTask):
    """
    Compute BERTopic model
    """

    kind = "compute_bertopic"

    def __init__(
        self,
        path_bertopic: Path,
        path_data: Path,
        col_id: str | None,
        col_text: str,
        parameters: BertTopicParamsModel,
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
        self.path_bertopic = path_bertopic.joinpath("bertopic")
        self.path_bertopic.mkdir(parents=True, exist_ok=True)
        self.path_data = path_data
        self.col_id = col_id
        self.col_text = col_text
        self.name = name or path_data.stem
        self.parameters = parameters
        self.existing_embeddings = existing_embeddings
        self.cols_embeddings = cols_embeddings
        self.timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        self.force_compute_embeddings = force_compute_embeddings
        with open(self.path_bertopic.joinpath("progress"), "w") as f:
            f.write("Initializing")

    def __call__(self):
        """
        Compute BERTopic model
        """
        # Load the data
        df = pd.read_parquet(self.path_data).fillna("")
        if self.col_text not in df.columns:
            raise ValueError(f"Column {self.col_text} not found in the data.")

        # Set the index if col_id is provided
        if self.col_id and self.col_id in df.columns:
            df.set_index(self.col_id, inplace=True)

        # Path for existing embeddings
        path_embeddings = self.path_bertopic.joinpath(
            slugify(f"bertopic_embeddings_{self.parameters.embeddings.model}.parquet")
        )
        # Path for 2D projection
        path_projection = self.path_bertopic.joinpath(
            slugify(f"bertopic_projection_{self.parameters.embeddings.model}.parquet")
        )

        # Check if existing embeddings are in the folder
        if not self.existing_embeddings and path_embeddings.exists():
            self.existing_embeddings = path_embeddings

        # Compute embeddings if not provided
        if not self.existing_embeddings or self.force_compute_embeddings:
            self.compute_embeddings(df, path_embeddings)
            self.compute_projection(path_embeddings, path_projection)
            self.existing_embeddings = path_embeddings

        # Compute projection if does not exist
        if not path_projection.exists():
            self.compute_projection(self.existing_embeddings, path_projection)

        # Load embeddings
        with open(self.path_bertopic.joinpath("progress"), "w") as f:
            f.write("Reading embeddings")
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
        with open(self.path_bertopic.joinpath("progress"), "w") as f:
            f.write("Fit BERTopic model")

        # Manage stopwords for cluster representation
        try:
            stopwords = self.get_stopwords()
            vectorizer_model = CountVectorizer(stop_words=stopwords)
        except ValueError:
            vectorizer_model = CountVectorizer()

        topic_model = BERTopic(
            language=self.parameters.language,
            vectorizer_model=vectorizer_model,
            nr_topics=self.parameters.nr_topics,
        )

        # Fit the BERTopic model
        topics, _ = topic_model.fit_transform(
            documents=df[self.col_text],
            embeddings=embeddings,
        )

        # Add outlier reduction
        if self.parameters.outlier_reduction:
            topics = topic_model.reduce_outliers(df[self.col_text], topics)

        # Add the topics to the DataFrame
        df["cluster"] = topics

        # Save the topics and documents informations
        topic_model.get_topic_info().to_csv(
            self.path_bertopic.joinpath(f"bertopic_topics_{self.name}.csv")
        )
        df["cluster"].to_csv(self.path_bertopic.joinpath(f"bertopic_clusters_{self.name}.csv"))
        # topic_model.get_document_info(df[self.col_text].tolist()).to_csv(
        #     self.path_bertopic.joinpath(f"bertopic_documents_{self.timestamp}.csv")
        # )

        self.path_bertopic.joinpath("progress").unlink(missing_ok=True)
        return None

    def get_stopwords(self) -> list[str]:
        """
        Get the stopwords for the specified language
        """
        if self.parameters.language not in stopwords.langs():
            raise ValueError(f"Unsupported language {self.parameters.language}")
        return list(stopwords.stopwords(self.parameters.language))

    def compute_embeddings(self, df: pd.DataFrame, path_embeddings: Path):
        """
        Compute the embeddings using the SBERT model
        """
        if self.parameters.embeddings.kind != "sentence_transformers":
            raise ValueError("Only sentence_transformers embeddings are supported for BERTopic.")
        with open(self.path_bertopic.joinpath("progress"), "w") as f:
            f.write("Computing embeddings")
        embeddings = ComputeSbert(
            texts=df[self.col_text],
            path_process=self.path_bertopic,
            model=self.parameters.embeddings.model,
            batch_size=32,
            min_gpu=1,
            path_progress=self.path_bertopic.joinpath("progress"),
        )()
        # save the embeddings to a file
        embeddings.to_parquet(path_embeddings)
        embeddings = embeddings.values

    def compute_projection(self, path_embeddings: Path, path_projection: Path) -> pd.DataFrame:
        """
        Reduce the dimensionality of the embeddings if needed.
        """
        try:
            reducer = cuml.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine")
            print("Using cuML for UMAP computation")
        except Exception:
            reducer = umap.UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine")
            print("Using standard UMAP for computation")
        embeddings = pd.read_parquet(path_embeddings)
        reduced_embeddings = reducer.fit_transform(embeddings)
        df_reduced = pd.DataFrame(reduced_embeddings, index=embeddings.index, columns=["x", "y"])
        df_reduced.to_parquet(path_projection)
