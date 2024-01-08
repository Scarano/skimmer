from skimmer.span_scorer import SpanScorer
import numpy as np


YELLOW_HIGHLIGHT_ATTRIBUTE = 'style="background-color: #ffff7f"'


class HTMLHighlighter:
    """
    This is a quick and dirty implementation that just adds HTML to the input text, and won't
    work well if input already contains HTML tags.
    TODO: finish the version that parses the input as HTML, which is started in branch html-highlighter
    """

    def __init__(self, scorer: SpanScorer, proportion: float = 0.5,
                 highlight_attribute: str = YELLOW_HIGHLIGHT_ATTRIBUTE):
        self.scorer = scorer
        self.proportion = proportion
        self.highlight_attribute = highlight_attribute

    def highlight(self, doc: str) -> str:

        spans = self.scorer(doc)

        highlight_pct = 100 * self.proportion

        threshold = np.percentile([span.score for span in spans], 100 - highlight_pct)

        highlighted = ''
        last_span_end = 0
        for span in spans:
            if span.score < threshold:
                highlighted += doc[last_span_end:span.end]
            else:
                highlighted += doc[last_span_end:span.start]
                highlighted += f'<span {self.highlight_attribute}>'
                highlighted += doc[span.start:span.end]
                highlighted += '</span>'
            last_span_end = span.end
        highlighted += doc[last_span_end:]

        return highlighted
