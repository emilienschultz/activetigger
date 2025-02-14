from pandas import DataFrame
from activetigger.datamodels import TsneModel, UmapModel, UserProjectionComputing
from activetigger.functions import compute_tsne, compute_umap


class Projections:
    """
    Manage projections
    """

    # TODO: Transform available type to dict[str, UserProjection]
    available: dict
    # TODO: Transform options type to ProjectionOptions (to create in datamodels).
    # Also, I don't see it use anywhere else, did I miss it?
    options: dict
    computing: list[UserProjectionComputing]

    def __init__(self, computing: list[UserProjectionComputing]) -> None:
        self.computing = computing
        self.available = {}
        self.options = {
            "umap": {
                "n_neighbors": 15,
                "min_dist": 0.1,
                "n_components": 2,
                "metric": ["cosine", "euclidean"],
            },
            "tsne": {
                "n_components": 2,
                "learning_rate": "auto",
                "init": "random",
                "perplexity": 3,
            },
        }

    def validate(self, method: str, params: dict) -> dict:
        if method == "umap":
            try:
                return {"func": compute_umap, "params": UmapModel(**params).__dict__}
            except Exception as e:
                return {"error": str(e)}
        if method == "tsne":
            try:
                return {"func": compute_tsne, "params": TsneModel(**params).__dict__}
            except Exception as e:
                return {"error": str(e)}
        return {"error": "Unknown method"}

    def current_computing(self):
        return [e.name for e in self.computing if e.kind == "projection"]

    def training(self) -> dict:
        """
        Currently under training
        """
        r = {e.user: e.method for e in self.computing if e.kind == "projection"}
        return r

    def add(self, element: UserProjectionComputing, results: DataFrame):
        """
        Add projection after computation
        """
        self.available[element.user] = {
            "data": results,
            "method": element.method,
            "params": element.params,
            "id": element.unique_id,
        }
