from typing import Optional
import re

from skimmer.span_scorer import SpanScorer
import numpy as np

NULL_PARAGRAPH_PATTERN = r'\w\b\w'
TEXT_PARAGRAPH_PATTERN = r'\n\s*\n'
HTML_PARAGRAPH_PATTERN = r'\n*<br/?>\n*'

class Abridger:

    def __init__(self, scorer: SpanScorer, keep: float = 0.5, max_spans: Optional[int] = None,
                 paragraph_pattern: str = NULL_PARAGRAPH_PATTERN, ellipsis_string = ' '):
        self.scorer = scorer
        self.keep = keep
        self.max_spans = max_spans
        self.paragraph_pattern = re.compile(paragraph_pattern)
        self.content_pattern = re.compile(r'\w+')
        self.ellipsis_string = ellipsis_string

    def abridge(self, doc: str) -> str:
        spans = self.scorer(doc)

        if self.max_spans is not None and len(spans) * self.keep > self.max_spans:
            keep_pct = 100 * self.max_spans / len(spans)
        else:
            keep_pct = 100 * self.keep

        threshold = np.percentile([span.score for span in spans], 100 - keep_pct)

        abridged = ''
        last_kept_span_end = 0
        for span, prev_span in zip(spans, [None] + spans[:-1]):
            if span.score >= threshold:
                omitted_text = doc[last_kept_span_end:span.start]
                if self.content_pattern.search(omitted_text):
                    abridged += self.ellipsis_string
                if m := self.paragraph_pattern.search(omitted_text):
                    abridged += m.group(0)
                abridged += doc[span.start:span.end]
                last_kept_span_end = span.end

        return abridged
