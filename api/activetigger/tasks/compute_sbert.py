import gc
import logging

import torch
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer
from torch import autocast

from activetigger.tasks.base_task import BaseTask


class ComputeSbert(BaseTask):
    """
    Compute sbert feature
    """

    kind = "compute_feature_sbert"

    def __init__(
        self, texts: Series, model: str = "all-mpnet-base-v2", batch_size: int = 32
    ):
        self.texts = texts
        self.model = model
        self.batch_size = batch_size

    def __call__(self) -> DataFrame:
        """
        Compute sbert embedding
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")  # Use CUDA
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")  # Fallback to CPU

        try:
            sbert = SentenceTransformer(self.model, device=str(device))
            sbert.max_seq_length = 512

            print("start computation")
            if device.type == "cuda":
                with autocast(device_type=str(device)):
                    emb = sbert.encode(
                        list(self.texts), device=str(device), batch_size=self.batch_size
                    )
            else:
                emb = sbert.encode(
                    list(self.texts), batch_size=self.batch_size, device=str(device)
                )
            emb = DataFrame(emb, index=self.texts.index)
            emb.columns = ["sb%03d" % (x + 1) for x in range(len(emb.columns))]
            logging.debug("computation end")
            return emb
        except Exception as e:
            logging.error(e)
            raise e
        finally:
            del sbert, self.texts
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
