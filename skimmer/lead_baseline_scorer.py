import numpy as np

from skimmer import logger
from skimmer.span_scorer import ScoredSpan, SpanScorer
from skimmer.parser import Parser


class LeadBaselineScorer(SpanScorer):
    def __init__(self, parser: Parser):
        """
        :param parser: `Parser` object to convert document strings in sequences of `DepParse`s.
        """
        self.parser = parser

    def __call__(self, doc: str) -> list[ScoredSpan]:
        # Use parser to divide doc into sentences (even tho we're not currently using the parses.)
        sent_parses = list(self.parser.parse(doc.strip()))

        # Scores simply go from 1 for the first sentence to 0 for the last.
        sent_scores = np.linspace(1.0, 0.0, len(sent_parses))

        return [ScoredSpan(start=sent_parse.start, end=sent_parse.end, score=score)
                for sent_parse, score in zip(sent_parses, sent_scores)]
