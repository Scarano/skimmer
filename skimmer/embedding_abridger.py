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

from skimmer.abridger import ScoredSpan
from skimmer.parser import Parser, DepParse
from skimmer.util import abbrev

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Method(Enum):
    SENTENCE_COMPLEMENT = 'sentence-complement'
    SENTENCE_SUMMARY_COMPLEMENT = 'sentence-summary-complement'
    SENTENCE_SUMMARY_SIMILARITY = 'sentence-summary-similarity'
    SENTENCE_SIMILARITY = 'sentence-similarity'
    SENTENCE_MEAN_OF_SAMPLES = 'sentence-mean-of-samples'
    SENTENCE_ITERATIVE = 'sentence-iterative'

    @classmethod
    def of(cls, s: str):
        for m in cls:
            if m.value == s:
                return m
        raise Exception(f"Unknown {cls.__name__} '{s}'")


# class EmbeddingSpace(ABC):
#     def encode(self, text: str) -> np.array:
#         pass

class EmbeddingAbridger:
    def __init__(self,
                 method: Method,
                 parser: Parser,
                 embedder: Callable[[str], np.array],
                 summarizer: Callable[[str], str]):
        self.parser = parser
        self.method = method
        self.embed = embedder
        self.summarize = summarizer

    def __call__(self, doc: str) -> list[ScoredSpan]:
        sent_parses = list(self.parser.parse(doc))

        if self.method == Method.SENTENCE_SUMMARY_COMPLEMENT:
            summary = self.summarize(doc)
            doc_target = self.embed(summary)
        else:
            doc_target = self.embed(doc)

        if self.method in [Method.SENTENCE_COMPLEMENT, Method.SENTENCE_SUMMARY_COMPLEMENT]:
            return self.score_sentence_complement(sent_parses, doc_target)
        elif self.method == Method.SENTENCE_SIMILARITY:
            return self.score_similarity(sent_parses, doc_target)
        elif self.method == Method.SENTENCE_MEAN_OF_SAMPLES:
            return self.score_mean_of_samples(sent_parses, doc_target)

    def score_sentence_complement(self, sent_parses: list[DepParse], doc_target: np.array) \
            -> list[ScoredSpan]:
        sent_scores = []
        for s, sent_parse in enumerate(sent_parses):
            logger.info("Getting embeddings with sentence %s omitted...", s)
            doc_minus_s = ' '.join(p.text for p in sent_parses[:s] + sent_parses[s+1:])
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

    # def score_iterative(self, sent_parses: list[DepParse], doc_target: np.array) \
    #         -> list[ScoredSpan]:
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


def scored_spans_as_html(doc: str, spans: list[ScoredSpan], f: IO):
    scores = np.array([span.score for span in spans])
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min())
    for i, span in enumerate(spans):
        if i > 0:
            f.write(doc[spans[i-1].end:span.start])
        red = (1.0 - norm_scores[i]) * 128 + 128
        green = norm_scores[i] * 128 + 128
        f.write(f'<span style="background-color: rgb({red},{green},127);">')
        f.write(f'<br>[{span.score:.4f}] {doc[span.start:span.end]}')
        f.write('</span>')


class OpenAISummarizer:
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

def score_to_html(doc, method, summary_override):
    parser = Parser('en')
    memory = joblib.Memory('cache', mmap_mode='c', verbose=0)
    embed = OpenAIEmbedding(memory=memory)
    if summary_override:
        summarize = lambda _: summary_override
    else:
        summarize = OpenAISummarizer(memory=memory)
    abridger = EmbeddingAbridger(method, parser, embed, summarize)
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
    parser.add_argument('--summary', type=str, default=None,
                        help='Path to target summary used to override generates summary (for testing)')
    parser.add_argument('--method', type=str, default='sentence-greedy',
                         choices=[m.value for m in Method],
                         help='Method to use for abridging')
    args = parser.parse_args()

    method = Method.of(parser.parse_args().method)

    with open(args.doc) as f:
        doc = f.read()
    if args.summary is None:
        summary = None
    else:
        with open(args.summary) as f:
            summary = f.read()
    score_to_html(doc, method, summary)

    return 0


if __name__=='__main__':
    logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)s: %(message)s')

    sys.exit(demo())

