import argparse
import logging
import math
import sys
import tempfile
import webbrowser
from enum import Enum
from typing import Callable, IO, Optional

import joblib
import numpy as np
from openai import OpenAI
import tiktoken

from skimmer.abridger import ScoredSpan, Abridger
from skimmer.parser import Parser, DepParse
from skimmer.util import abbrev, batched

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Method(Enum):
    SENTENCE_COMPLEMENT = 'sentence-complement'
    SENTENCE_SUMMARY_COMPLEMENT = 'sentence-summary-complement'
    SENTENCE_SIMILARITY = 'sentence-similarity'
    SENTENCE_MEAN_OF_SAMPLES = 'sentence-mean-of-samples'
    SENTENCE_ITERATIVE = 'sentence-iterative'

    @classmethod
    def default(cls) -> 'Method':
        return Method.SENTENCE_SUMMARY_COMPLEMENT

    @classmethod
    def of(cls, s: str):
        for m in cls:
            if m.value == s:
                return m
        raise Exception(f"Unknown {cls.__name__} '{s}'")


class EmbeddingAbridger(Abridger):
    def __init__(self,
                 method: Method,
                 chunk_size: int,
                 parser: Parser,
                 embedder: Callable[[str], np.array],
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
        sent_parses = list(self.parser.parse(doc))

        scored_spans = []
        for chunk in batched(sent_parses, self.chunk_size):
            chunk_str = '\n'.join(sent.text for sent in chunk)

            if self.method in [Method.SENTENCE_SUMMARY_COMPLEMENT]:
                summary = self.summarize(chunk_str)
                chunk_target = self.embed(summary)
            else:
                chunk_target = self.embed(chunk_str)

            if self.method in [Method.SENTENCE_COMPLEMENT, Method.SENTENCE_SUMMARY_COMPLEMENT]:
                chunk_scores = self.score_sentence_complement(chunk, chunk_target)
            elif self.method == Method.SENTENCE_SIMILARITY:
                chunk_scores = self.score_similarity(chunk, chunk_target)
            elif self.method == Method.SENTENCE_MEAN_OF_SAMPLES:
                chunk_scores = self.score_mean_of_samples(chunk, chunk_target)
            else:
                raise Exception(f"Method not implemented: {self.method}")

            # Noramlize to chunk-specific z-score:
            # (TODO: Should this really be necessary? Is there a better way to calculating the
            # scores that doesn't need normalization?)
            scores = np.array([span.score for span in chunk_scores])
            mean = np.mean(scores)
            std = np.std(scores)
            scored_spans += [span.copy(update={"score": (span.score - mean)/std})
                             for span in chunk_scores]

        return scored_spans

    def score_sentence_complement(self, sent_parses: list[DepParse], doc_target: np.array) \
            -> list[ScoredSpan]:
        """
        The score of sentence `i` is based on the cosine similarity between the target embedding
        (`doc_target`) and the embedding we get from omitting sentence `i`.
        That number is typically close to 1. To turn that into a distance, we subtract it from 1.

        Because the resulting distance is typically very small, and can vary in order of magnitude
        we currently take the log of that.
        TODO: Look at score distributions to decide if it really makes sense to take the log.
        """
        sent_scores = []
        for s, sent_parse in enumerate(sent_parses):
            logger.info("Getting embeddings with sentence %s omitted...", s)
            doc_minus_s = '\n'.join(p.text for p in sent_parses[:s] + sent_parses[s+1:])
            doc_minus_s_embedding = self.embed(doc_minus_s)
            sent_scores.append(math.log(1.0 - np.dot(doc_target, doc_minus_s_embedding)))

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]

    def score_similarity(self, sent_parses: list[DepParse], doc_target: np.array) \
            -> list[ScoredSpan]:
        sent_scores = []
        for s, sent_parse in enumerate(sent_parses):
            s_embedding = self.embed(sent_parse.text)
            sent_scores.append(np.dot(doc_target, s_embedding))

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]

    def score_mean_of_samples(self, sent_parses: list[DepParse], doc_target: np.array) \
            -> list[ScoredSpan]:

        np.random.seed(1)

        local_scores = []
        for s, sent_parse in enumerate(sent_parses):
            logger.info("Getting embeddings with sentence %s omitted...", s)
            doc_minus_s = ' '.join(p.text for p in sent_parses[:s] + sent_parses[s+1:])
            doc_minus_s_embedding = self.embed(doc_minus_s)
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
            resampled_embedding = self.embed(resampled_doc)

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


class OpenAISummarizer:
    """
    Summarization function (in the sense that it is __call__-able) that uses OpenAI's chat
    interface with a summarization prompt.
    """

    SUMMARIZE_PROMPT = """
        As a professional abridger, write a slightly shortened version of the provided text.
        Your version should include all the main ideas and essential information, but eliminate extraneous language, less-important points, redundant information, and redundant examples.
        It should preserve the style and jargon of the provided text.
        Rely strictly on the provided text, without including external information.
    """
        # Your version should be about three quarters of the length of the provided text.
    MAX_TOKENS = 4000  # TODO: should be based on model choice, not a constant

    client = OpenAI()  # TODO: should probably be passed in to constructor

    def __init__(self, model: str = 'gpt-3.5-turbo',
                 memory: Optional[joblib.Memory] = None):
        """
        :param model: OpenAI model to use for summarization
        :param memory: joblib Memory object to use for caching. If None, no caching will be done.
        """
        self.model = model
        try:
            self.encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            raise Exception(f"No tiktoken encoding found for model {model}")

        self.summarize_func = lambda model, encoding, prompt, text: \
            OpenAISummarizer.uncached_summarize(
                model, encoding, prompt, text)
        if memory:
            self.summarize_func = memory.cache(self.summarize_func)

    @staticmethod
    def uncached_summarize(model, encoding, prompt, text: str) -> str:
        text = text.strip()

        num_tokens = len(encoding.encode(text))
        if num_tokens > OpenAISummarizer.MAX_TOKENS:
            raise Exception(
                f"OpenAISummarizer does not support more than {OpenAISummarizer.MAX_TOKENS} "
                f"tokens at a time. Provided text ({text[:30]}...) has {num_tokens} tokens.")

        response = OpenAISummarizer.client.chat.completions.create(
            model=model, max_tokens=num_tokens, temperature=0,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ]
        )
        # TODO Check finish_reason / errors
        summary = response.choices[0].message.content.strip()
        logger.debug("Summarization response for %s: %s",
                     abbrev(text, 40),
                     abbrev( repr(response), 1000))

        return summary

    def __call__(self, text: str) -> str:
        return self.summarize_func(
            self.model, self.encoding, OpenAISummarizer.SUMMARIZE_PROMPT, text)


class OpenAIEmbedding:
    client = OpenAI()  # TODO: should probably be passed in to constructor

    def __init__(self, model: str = 'text-embedding-ada-002',
                 memory: Optional[joblib.Memory] = None):
        """
        :param model: OpenAI model to use for embedding
        :param memory: joblib Memory object to use for caching. If None, no caching will be done.
        """
        self.model = model
        self.embed_func = lambda model, text: OpenAIEmbedding.uncached_embed(model, text)
        if memory:
            self.embed_func = memory.cache(self.embed_func)

    @staticmethod
    def uncached_embed(model, text):
        response = OpenAIEmbedding.client.embeddings.create(model=model, input=text)
        # TODO error checking
        return np.array(response.data[0].embedding)

    def __call__(self, text: str) -> np.array:
        return self.embed_func(self.model, text)


def scored_spans_as_html(doc: str, spans: list[ScoredSpan], f: IO):
    scores = np.array([span.score for span in spans])
    colors = np.array([[255, 127, 127],   # Red for lowest score
                       [255, 255, 255],   # White for median
                       [127, 255, 127]])  # Green for highest
    min_score = np.min(scores)
    median_score = np.median(scores)
    max_score = np.max(scores)
    for i, span in enumerate(spans):
        if i > 0:
            f.write(doc[spans[i-1].end:span.start])
        rgb = [str(int(np.interp(scores[i], [min_score, median_score, max_score], col)))
               for col in colors.T]
        f.write(f'<span style="background-color: rgb({",".join(rgb)});">')
        f.write(f'<br>[{span.score:.3f}] {doc[span.start:span.end]}')
        f.write('</span>')


def score_to_html(doc, method, chunk_size, summary_override):
    """
    Do a demo run of the EmbeddingAbridger on a single doc.
    :param doc: Document to abridge
    :param method: Method to use for abridging
    :param chunk_size: Number of sentences to abridge as a unit
    :param summary_override: If not None, this string will be used as the summary instead of
        generating one. Only for testing purposes. And only for when doc has fewer than
        `chunk_size` sentences.
    """
    parser = Parser('en')
    memory = joblib.Memory('cache', mmap_mode='c', verbose=0)
    embed = OpenAIEmbedding(memory=memory)
    if summary_override:
        summarize = lambda _: summary_override
    else:
        summarize = OpenAISummarizer(memory=memory)
    abridger = EmbeddingAbridger(method, chunk_size, parser, embed, summarize)
    spans = abridger(doc)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        print(f"Saving HTML output to: {f.name}")
        scored_spans_as_html(doc, spans, f)
        webbrowser.open('file://' + f.name, new=2)
        # input('Press enter to complete and delete temp file.')


def demo():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Abridger')
    parser.add_argument('doc', type=str, help='Path to document to abridge')
    parser.add_argument('--method', type=str, default=Method.default().value,
                         choices=[m.value for m in Method],
                         help='Method to use for abridging')
    parser.add_argument('--chunk-size', type=int, default=20,
                        help='Maximum number of sentences to abridge at once')
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to target summary used to override generates summary (for testing)')
    args = parser.parse_args()

    method = Method.of(parser.parse_args().method)

    with open(args.doc) as f:
        doc = f.read()
    if args.summary is None:
        summary = None
    else:
        with open(args.summary) as f:
            summary = f.read()
    score_to_html(doc, method, args.chunk_size, summary)

    return 0


if __name__=='__main__':
    logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)s: %(message)s')

    sys.exit(demo())

