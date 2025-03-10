import gc
import logging
import math
import os
from pathlib import Path

import numpy as np
import torch
from pandas import DataFrame, Series
from sentence_transformers import SentenceTransformer

from activetigger.tasks.base_task import BaseTask


class ComputeSbert(BaseTask):
    """
    Compute sbert feature
    """

    kind = "compute_feature_sbert"

    def __init__(
        self,
        texts: Series,
        path_process: Path,
        model: str = "all-mpnet-base-v2",
        batch_size: int = 32,
        min_gpu: int = 6,
    ):
        super().__init__()
        self.texts = texts
        self.model = model
        self.batch_size = batch_size
        self.min_gpu = min_gpu
        self.path_process = path_process

    def __call__(self) -> DataFrame:
        """
        Compute sbert embedding
        """

        # test the data
        if self.texts.isnull().sum() > 0:
            raise ValueError(
                "There are missing values in the input data, so we can't proceed"
            )

        if torch.cuda.is_available():
            if (
                torch.cuda.get_device_properties(0).total_memory / (1024**3)
                > self.min_gpu
            ):
                device = torch.device("cuda")  # Use CUDA
            else:
                print("Not enough GPU memory, fallback to CPU")
                device = torch.device("cpu")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")  # Fallback to CPU

        try:
            sbert = SentenceTransformer(self.model, device=str(device))
            sbert.max_seq_length = 512

            print("start computation")
            embeddings = []
            total_batches = math.ceil(len(self.texts) / self.batch_size)
            # if device.type == "cuda":
            #     with autocast(device_type=str(device)):
            #         emb = sbert.encode(
            #             list(self.texts),
            #             device=str(device),
            #             batch_size=self.batch_size,
            #         )
            # else:
            for i, start in enumerate(range(0, len(self.texts), self.batch_size), 1):
                batch_texts = list(self.texts.iloc[start : start + self.batch_size])
                embeddings.append(sbert.encode(batch_texts, device=str(device)))

                # manage progress
                progress_percent = (i / total_batches) * 100
                with open(self.path_process.joinpath(self.unique_id), "w") as f:
                    f.write(str(round(progress_percent, 1)))
                print(progress_percent)

            # shape the data
            emb = DataFrame(
                np.vstack(embeddings),
                index=self.texts.index,
                columns=["sb%03d" % (x + 1) for x in range(len(embeddings[0][0]))],
            )
            logging.debug("computation end")
            os.remove(self.path_process.joinpath(self.unique_id))
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
