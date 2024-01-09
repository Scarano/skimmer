import logging
import re
from itertools import combinations, groupby
import numpy as np
import numpy.typing as npt
from typing import Callable, IO

from skimmer.span_scorer import SpanScorer, ScoredSpan
from skimmer.parser import Parser, StanzaParser, DepParse
from skimmer.util import equal_spans, IndexedEnum


class Method(IndexedEnum):
    SENTENCE_SUMMARY_MATCHING = 'sentence-summary-matching'
    CLAUSE_SUMMARY_MATCHING = 'clause-summary-matching'

    @classmethod
    def default(cls) -> 'Method':
        return Method.SENTENCE_SUMMARY_MATCHING


class SummaryMatchingScorer(SpanScorer):
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
        summary_embed = self.embed([parse.text for parse in summary_parses
                                               if len(parse.text) >= 4]) # exclude occasional debris
        for i in range(len(summary_embed)):
            # For each sentence i, add cosine similarity between this sentence's embedding and
            # sent_embeddings[i] to scores[i].
            # print(sent_embeddings @ summary_embed[i])
            scores = np.maximum(scores, sent_embeddings @ summary_embed[i])
            # scores += sent_embeddings @ summary_embed[i]

        # scores /= len(sent_parses)

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end,
                           text=sent_parse.text,
                           score=score)
                for sent_parse, score in zip(sent_parses, scores)]


class SummaryMatchingClauseScorer:
    MIN_MODIFIER_LENGTH = 3
    MODIFIER_TYPES = \
        'acl advcl advmod amod appos det iobj nmod nummod nummod obl reparandum vocative '.split()
    MODIFIER_PATTERN = re.compile('|'.join(MODIFIER_TYPES) + '.*', re.IGNORECASE)

    def __init__(self,
                 parser: StanzaParser,
                 embed: Callable[[list[str]], npt.NDArray[np.float_]],
                 summarize: Callable[[str], str],
                 length_penalty: float = 1.0,
                 max_removals: int = 3):
        self.parser = parser
        self.embed = embed
        self.summarize = summarize
        self.length_penalty = length_penalty
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
            # set var_scores to max of var_scores and `sent_embeddings @ summary_embed[i]`
            var_scores = np.maximum(var_scores, sent_embeddings @ summary_embed[i])
        # var_scores /= len(summary_embed)

        var_lengths = np.array([float(len(p)) for _, p in variations])
        var_scores *= var_lengths ** -self.length_penalty

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
                     if SummaryMatchingClauseScorer.MODIFIER_PATTERN.fullmatch(r)
                     and len(parse.constituents[i]) >= SummaryMatchingClauseScorer.MIN_MODIFIER_LENGTH]
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

