from typing import Optional, List

import joblib
import numpy as np
from numpy import typing as npt
from openai import OpenAI, RateLimitError

from skimmer import logger
from skimmer.util import batched, with_retry


class OpenAIEmbedding:
    client = OpenAI()  # TODO: should probably be passed in to constructor

    def __init__(self, model: str = 'text-embedding-ada-002',
                 memory: Optional[joblib.Memory] = None):
        """
        :param model: OpenAI model to use for embedding
        :param memory: joblib Memory object to use for caching. If None, no caching will be done.
        """
        self.model = model
        self.embed_func = lambda mdl, texts: OpenAIEmbedding.uncached_embed(mdl, texts)
        if memory:
            self.embed_func = memory.cache(self.embed_func)

    @staticmethod
    def uncached_embed(model: str, texts: list[str]) -> npt.NDArray[np.float_]:
        results: List[List[float]] = []
        for batch in batched(texts, 100):
            logger.debug("Getting embeddings for %s strings", len(batch))
            # logger.debug("Getting embeddings for %s", batch)
            response = with_retry(
                lambda: OpenAIEmbedding.client.embeddings.create(model=model, input=batch),
                RateLimitError,
                5,
                30)
            # TODO error checking
            results.extend(d.embedding for d in response.data)
        return np.array(results)

    def __call__(self, texts: list[str]) -> npt.NDArray[np.float_]:
        return self.embed_func(self.model, texts)
