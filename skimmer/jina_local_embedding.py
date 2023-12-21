from typing import Optional, List

import joblib
import numpy as np
from numpy import typing as npt
from transformers import AutoModel

from skimmer import logger
from skimmer.util import batched


class JinaLocalEmbedding:

    SMALL_MODEL = 'jinaai/jina-embeddings-v2-small-en'

    def __init__(self, model_name: str = SMALL_MODEL,
                 memory: Optional[joblib.Memory] = None):
        """
        :param model_name: HuggingFace model name
        :param memory: joblib Memory object to use for caching. If None, no caching will be done.
        """
        self.model_name = model_name
        # trust_remote_code is needed to use the encode method
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.embed_func = lambda _, texts: JinaLocalEmbedding.uncached_embed(self.model, texts)
        if memory:
            self.embed_func = memory.cache(self.embed_func)

    @staticmethod
    def uncached_embed(model, texts: list[str]) -> npt.NDArray[np.float_]:
        results: List[List[float]] = []
        for batch in batched(texts, 100):
            logger.debug("Getting embeddings for %s strings", len(batch))
            # logger.debug("Getting embeddings for %s", batch)
            embeddings = model.encode(batch)
            results.extend(embeddings)
        return np.array(results)

    def __call__(self, texts: list[str]) -> npt.NDArray[np.float_]:
        # Include model_name, even tho it's ignored, so that it is part of the joblib.Memory
        # cache key, and different models' results are kept separate.
        return self.embed_func(self.model_name, texts)
