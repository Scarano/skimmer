import logging
import math
from typing import Callable

import numpy as np
import numpy.typing as npt

from skimmer import logger
from skimmer.span_scorer import ScoredSpan, SpanScorer
from skimmer.parser import DepParse, Parser
from skimmer.util import IndexedEnum, batched_adaptive


class Method(IndexedEnum):
    SENTENCE_COMPLEMENT = 'sentence-complement'
    SENTENCE_SUMMARY_COMPLEMENT = 'sentence-summary-complement'
    SENTENCE_SIMILARITY = 'sentence-similarity'
    SENTENCE_MEAN_OF_SAMPLES = 'sentence-mean-of-samples'
    SENTENCE_ITERATIVE = 'sentence-iterative'

    @classmethod
    def default(cls) -> 'Method':
        return Method.SENTENCE_SUMMARY_COMPLEMENT



class EmbeddingScorer(SpanScorer):
    def __init__(self,
                 method: Method,
                 chunk_size: int,
                 parser: Parser,
                 embedder: Callable[[list[str]], npt.NDArray[np.float_]],
                 summarizer: Callable[[str], str]):
        """
        :param method: Method to use for abridging
        :param chunk_size: Number of sentences to abridge as a unit
        :param parser: `Parser` object to convert document strings in sequences of `DepParse`s.
        :param embedder: Returns embedding of input string.
        :param summarizer: Returns summary of input string.
        """
        self.method = method
        self.chunk_size = chunk_size
        self.parser = parser
        self.embed = embedder
        self.summarize = summarizer

    def __call__(self, doc: str) -> list[ScoredSpan]:
        # Use parser to divide doc into sentences (even tho we're not currently using the parses.)
        sent_parses = list(self.parser.parse(doc.strip()))

        scored_spans = []
        for chunk in batched_adaptive(sent_parses, self.chunk_size):
            chunk_str = '\n'.join(sent.text for sent in chunk)

            if self.method in [Method.SENTENCE_SUMMARY_COMPLEMENT]:
                summary = self.summarize(chunk_str)
                chunk_target = self.embed([summary])[0]
            else:
                chunk_target = self.embed([chunk_str])[0]

            if self.method in [Method.SENTENCE_COMPLEMENT, Method.SENTENCE_SUMMARY_COMPLEMENT]:
                chunk_scores = self.score_sentence_complement(chunk, chunk_target)
            elif self.method == Method.SENTENCE_SIMILARITY:
                chunk_scores = self.score_similarity(chunk, chunk_target)
            elif self.method == Method.SENTENCE_MEAN_OF_SAMPLES:
                chunk_scores = self.score_mean_of_samples(chunk, chunk_target)
            else:
                raise Exception(f"Method not implemented: {self.method}")

            scored_spans += chunk_scores

            # # Noramlize to chunk-specific z-score:
            # # (TODO: Should this really be necessary? Is there a better way to calculating the
            # # scores that doesn't need normalization?)
            # scores = np.array([span.score for span in chunk_scores])
            # mean = np.mean(scores)
            # std = np.std(scores)
            # scored_spans += [span.copy(update={"score": (span.score - mean)/std})
            #                  for span in chunk_scores]

        return scored_spans

    def score_sentence_complement(self,
                                  sent_parses: list[DepParse], doc_target: npt.NDArray[np.float_]) \
            -> list[ScoredSpan]:
        """
        The score of the sentence at `sent_parses[i]` is based on the cosine similarity between
        the target embedding (`doc_target`) and the embedding we get from omitting sentence `i`.
        That number is typically close to 1. To turn that into a distance, we subtract it from 1.

        Because the resulting distance is typically very small, and can vary in order of magnitude
        we currently take the log of that.
        TODO: Look at score distributions to decide if it really makes sense to take the log.
        """
        ablated_docs = []
        for s, sent_parse in enumerate(sent_parses):
            ablated_docs.append('\n'.join(p.text for p in sent_parses[:s] + sent_parses[s+1:]))
        ablated_embeddings = self.embed(ablated_docs)
        sent_scores = [math.log(1.0 - np.dot(doc_target, ablated_embedding))
                       for ablated_embedding in ablated_embeddings]

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]

    def score_similarity(self, sent_parses: list[DepParse], doc_target: npt.NDArray[np.float_]) \
            -> list[ScoredSpan]:
        sent_scores = []
        for s, sent_parse in enumerate(sent_parses):
            s_embedding = self.embed([sent_parse.text])[0]
            sent_scores.append(np.dot(doc_target, s_embedding))

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]

    def score_mean_of_samples(self, sent_parses: list[DepParse], doc_target: npt.NDArray[np.float_]) \
            -> list[ScoredSpan]:
        """ Experimental, does not work well so far. """

        np.random.seed(1)

        local_scores = []
        for s, sent_parse in enumerate(sent_parses):
            logger.info("Getting embeddings with sentence %s omitted...", s)
            doc_minus_s = ' '.join(p.text for p in sent_parses[:s] + sent_parses[s+1:])
            doc_minus_s_embedding = self.embed([doc_minus_s])[0]
            local_scores.append(np.dot(doc_target, doc_minus_s_embedding))

        # compute rank_scores as rank order of values in sent_scores
        rank_scores = np.argsort(local_scores)
        print(rank_scores)

        # Compress to probabilities in range [.25, .75]
        sample_prob = 1.0 - (rank_scores / len(local_scores) * 0.5 + 0.25)
        print(sample_prob)

        num_samples = 10

        scores = np.zeros(shape=(num_samples,))
        selected = np.full((len(sent_parses), num_samples), False, dtype=bool)

        for t in range(num_samples):
            # Sample from sent_parses so that sent_parses[i] has probability sample_prob[i] of being selected.
            random_values = np.random.rand(len(sent_parses))
            selected[:, t] = random_values < sample_prob

            resampled_doc = ' '.join(p.text for i, p in enumerate(sent_parses) if selected[i, t])
            resampled_embedding = self.embed([resampled_doc])[0]

            scores[t] = np.dot(doc_target, resampled_embedding)

        # create array with shape (len(sent_parses), len(scores)) by broadcasting scores vertically:

        incl_scores = np.ma.masked_array(np.tile(scores, (len(sent_parses), 1)),
                                         mask=~selected)
        # print(incl_scores)
        excl_scores = np.ma.masked_array(np.tile(scores, (len(sent_parses), 1)),
                                         mask=selected)
        # print(excl_scores)
        sent_scores = incl_scores.mean(axis=1) - excl_scores.mean(axis=1)
        # print(sent_scores)

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]

