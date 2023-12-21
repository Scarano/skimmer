from typing import Optional

from skimmer.span_scorer import SpanScorer
import numpy as np


class Abridger:
    def __init__(self, scorer: SpanScorer, keep: float = 0.5, max_spans: Optional[int] = None):
        self.scorer = scorer
        self.keep = keep
        self.max_spans = max_spans

    def abridge(self, doc: str) -> str:
        spans = self.scorer(doc)

        if self.max_spans is not None and len(spans) * self.keep > self.max_spans:
            keep_pct = 100 * self.max_spans / len(spans)
        else:
            keep_pct = 100 * self.keep

        threshold = np.percentile([span.score for span in spans], 100 - keep_pct)

        abridged = ' '.join(doc[span.start:span.end]
                            for span in spans
                            if span.score >= threshold)

        return abridged
