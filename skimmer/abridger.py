from skimmer.span_scorer import SpanScorer
import numpy as np


class Abridger:
    def __init__(self, scorer: SpanScorer, keep: float = 0.5):
        self.scorer = scorer
        self.keep = keep

    def abridge(self, doc: str) -> str:
        spans = self.scorer(doc)

        threshold = np.percentile([span.score for span in spans], 100 - 100 * self.keep)

        abridged = ' '.join(doc[span.start:span.end]
                            for span in spans if span.score > threshold)

        return abridged
