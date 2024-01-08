from typing import Optional
import re

from skimmer.span_scorer import SpanScorer
import numpy as np

NULL_PARAGRAPH_PATTERN = r'\w\b\w'  # match nothing; do not detect paragraph breaks
TEXT_PARAGRAPH_PATTERN = r'\n\s*\n'
HTML_PARAGRAPH_PATTERN = r'\n*<br/?>\n*'

class Abridger:
    """
    Use specified SpanScorer to remove less important/relevant spans.
    """

    def __init__(self, scorer: SpanScorer, keep: float = 0.5, max_spans: Optional[int] = None,
                 paragraph_pattern: str = NULL_PARAGRAPH_PATTERN, ellipsis_string = ' '):
        """
        :param scorer: SpanScorer to use to score spans
        :param keep: proportion of spans to keep (0 to 1)
        :param max_spans: maximum number of spans to keep (used if non-null, and less than
            keep * number of spans)
        :param paragraph_pattern: regular expression pattern used to detect a paragraph breaks.
            If an omitted section contains at least one match to this pattern, the first match
            is re-inserted so that the preceding and following (kept) spans are separated by
            exactly one paragraph break.
        :param ellipsis_string: string to insert in place of omitted text.
        """
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
