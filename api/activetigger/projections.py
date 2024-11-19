from activetigger.datamodels import TsneModel, UmapModel
from activetigger.functions import compute_tsne, compute_umap


class Projections:
    """
    Manage projections
    """

    available: dict
    options: dict
    training: dict

    def __init__(self) -> None:
        self.training: dict = {}
        self.available: dict = {}
        self.options: dict = {
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
