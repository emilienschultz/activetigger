from pandas import DataFrame

from activetigger.tasks.base_task import BaseTask

# accelerate UMAP
try:
    import cuml  # type: ignore[import-not-found]

    CUMl_AVAILABLE = True
except ImportError:
    print("CuML not installed")
    CUMl_AVAILABLE = False

import pandas as pd
import umap
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from activetigger.datamodels import TsneModel, UmapModel


class ComputeProjection(BaseTask):
    """
    Compute projection
    """

    kind = "projection"

    def __init__(self, kind: str, features: DataFrame, params: TsneModel | UmapModel):
        self.kind = kind
        self.features = features
        self.params = params

    def __call__(self):
        """
        Compute projection
        """
        if self.kind == "umap":
            return self.compute_umap()
        elif self.kind == "tsne":
            return self.compute_tsne()
        else:
            raise ValueError(f"Unknown kind {self.kind}")

    def compute_umap(self) -> DataFrame:
        """
        Compute UMAP
        """
        scaled_features = StandardScaler().fit_transform(self.features)

        # Check if cuML is available for GPU acceleration
        try:
            reducer = cuml.UMAP(**self.params.__dict__)
            print("Using cuML for UMAP computation")
        except Exception:
            reducer = umap.UMAP(**self.params.__dict__)
            print("Using standard UMAP for computation")

        reduced_features = reducer.fit_transform(scaled_features)
        df = pd.DataFrame(reduced_features, index=self.features.index)
        df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
        return df_scaled

    def compute_tsne(self) -> DataFrame:
        """
        Compute TSNE
        """
        scaled_features = StandardScaler().fit_transform(self.features)
        reduced_features = TSNE(**self.params.__dict__).fit_transform(scaled_features)
        df = pd.DataFrame(reduced_features, index=self.features.index)
        df_scaled = 2 * (df - df.min()) / (df.max() - df.min()) - 1
        return df_scaled
