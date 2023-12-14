import argparse
import logging
import re
import sys
import tempfile
import webbrowser
from enum import Enum
from itertools import combinations, groupby
import numpy as np
import numpy.typing as npt
from typing import Callable, IO

import joblib

from skimmer.abridger import Abridger, ScoredSpan
from skimmer.embedding_abridger import OpenAIEmbedding, OpenAISummarizer
from skimmer.parser import Parser, StanzaParser, DepParse, RightBranchingParser
from skimmer.util import equal_spans

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class Method(Enum):
    SENTENCE_SUMMARY_MATCHING = 'sentence-summary-matching'
    CLAUSE_SUMMARY_MATCHING = 'clause-summary-matching'

    @classmethod
    def default(cls) -> 'Method':
        return Method.CLAUSE_SUMMARY_MATCHING

    @classmethod
    def of(cls, s: str):
        for m in cls:
            if m.value == s:
                return m
        raise Exception(f"Unknown {cls.__name__} '{s}'")


class SummaryMatchingAbridger(Abridger):
    def __init__(self,
                 parser: Parser,
                 embed: Callable[[list[str]], npt.NDArray[np.float_]],
                 summarize: Callable[[str], str]):
        self.parser = parser
        self.embed = embed
        self.summarize = summarize

    def __call__(self, doc: str) -> list[ScoredSpan]:
        sent_parses = list(self.parser.parse(doc))
        sent_embeddings = self.embed([sent.text for sent in sent_parses])

        summary = self.summarize(doc)

        scores = np.zeros(len(sent_parses))
        summary_parses = self.parser.parse(summary)
        summary_embed = self.embed([parse.text for parse in summary_parses])
        for i in range(len(summary_embed)):
            # For each sentence i, add cosine similarity between this sentence's embedding and
            # sent_embeddings[i] to scores[i].
            # print(sent_embeddings @ summary_embed[i])
            scores += sent_embeddings @ summary_embed[i]

        scores /= len(sent_parses)

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end,
                           text=sent_parse.text,
                           score=score)
                for sent_parse, score in zip(sent_parses, scores)]


class SummaryMatchingClauseAbridger:
    MIN_MODIFIER_LENGTH = 3
    MODIFIER_TYPES = \
        'acl advcl advmod amod appos det iobj nmod nummod nummod obl reparandum vocative '.split()
    MODIFIER_PATTERN = re.compile('|'.join(MODIFIER_TYPES) + '.*', re.IGNORECASE)

    def __init__(self,
                 parser: StanzaParser,
                 embed: Callable[[list[str]], npt.NDArray[np.float_]],
                 summarize: Callable[[str], str],
                 max_removals: int = 3):
        self.parser = parser
        self.embed = embed
        self.summarize = summarize
        self.max_removals = max_removals

    def __call__(self, doc: str) -> list[ScoredSpan]:
        sent_parses = list(self.parser.parse(doc))
        variations = [(s, v)
                      for s, p in enumerate(sent_parses)
                      for v in self.generate_sentence_variations(p)]
        # for (s, v) in variations:
        #     logger.info("%d: %s", s, v)
        #     logger.info("%s", self.substring(doc, sent_parses[s], v))
        sent_embeddings = self.embed([self.substring(doc, sent_parses[s], v)
                                      for s, v in variations])

        summary = self.summarize(doc)

        var_scores = np.zeros(len(variations))
        summary_parses = self.parser.parse(summary)
        summary_embed = self.embed([parse.text for parse in summary_parses])
        # For each sentence variation i, add cosine similarity between this sentence's embedding
        # and sent_embeddings[i] to scores[i].
        for i in range(len(summary_embed)):
            # print(sent_embeddings @ summary_embed[i])
            var_scores += sent_embeddings @ summary_embed[i]
        var_scores /= len(summary_embed)

        scored_spans = []
        for s, var_pairs in groupby(zip(variations, var_scores), key=lambda pair: pair[0][0]):
            token_max_scores = np.zeros(len(sent_parses[s]))
            for (_, v), score in var_pairs:
                # logger.info("%f.3: %s", score, self.substring(doc, sent_parses[s], v))
                for i in v:
                    if score > token_max_scores[i]:
                        token_max_scores[i] = score

            for start, end, score in equal_spans(token_max_scores):
                scored_spans.append(ScoredSpan(start=sent_parses[s].spans[start][0],
                                               end=sent_parses[s].spans[end][1],
                                               score=score))

        return scored_spans

    def generate_sentence_variations(self, parse: DepParse) -> list[list[int]]:
        modifiers = [i
            for i, r in enumerate(parse.relations)
            if SummaryMatchingClauseAbridger.MODIFIER_PATTERN.fullmatch(r)
                and len(parse.constituents[i]) >= SummaryMatchingClauseAbridger.MIN_MODIFIER_LENGTH]
        top_modifiers = sorted(modifiers, key=lambda i: -len(parse.constituents[i]))[:self.max_removals]

        # combos is power set of modifiers to be considered for removal
        combos = [list(comb) for k in range(len(top_modifiers) + 1)
                             for comb in combinations(top_modifiers, k)]

        variations = set()
        for combo in combos:
            mask = set.union(set(), *(parse.constituents[i] for i in combo))
            included_tokens = [i for i in range(len(parse)) if i not in mask]
            variations.add(tuple(included_tokens))

        return [list(v) for v in sorted(variations)]

    @staticmethod
    def substring(doc, sent: DepParse, included_tokens: list[int]) -> str:
        return ' '.join(doc[start:end]
                        for start, end in sent.subspans(included_tokens))

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
            f.write(doc[spans[i-1].end:span.start].replace('\n\n', '\n<br/><br/>\n'))
        rgb = [str(int(np.interp(scores[i], [min_score, median_score, max_score], col)))
               for col in colors.T]
        span_text = doc[span.start:span.end].replace('\n\n', '\n<br/><br/>\n')
        f.write(f'<span style="background-color: rgb({",".join(rgb)});">')
        f.write(f'{span_text} [{span.score:.3f}] ')
        f.write('</span>')

def score_to_html(doc, method):
    """
    Do a demo run of the EmbeddingAbridger on a single doc.
    :param doc: Document to abridge
    :param method: Method to use for abridging
    """
    memory = joblib.Memory('cache', mmap_mode='c', verbose=0)
    embed = OpenAIEmbedding(memory=memory)
    summarize = OpenAISummarizer(memory=memory)
    if method == Method.SENTENCE_SUMMARY_MATCHING:
        parser = RightBranchingParser('en')
        abridger = SummaryMatchingAbridger(parser, embed, summarize)
    elif method == Method.CLAUSE_SUMMARY_MATCHING:
        parser = StanzaParser('en')
        abridger = SummaryMatchingClauseAbridger(parser, embed, summarize)
    else:
        raise Exception(f"invalid method: {method}")

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
    args = parser.parse_args()

    method = Method.of(parser.parse_args().method)

    with open(args.doc) as f:
        doc = f.read()
    score_to_html(doc, method)

    return 0


if __name__=='__main__':
    logging.basicConfig(level=logging.WARN, format='%(asctime)s %(levelname)s: %(message)s')

    sys.exit(demo())

