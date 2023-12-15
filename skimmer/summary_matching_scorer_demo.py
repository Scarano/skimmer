import argparse
import tempfile
import webbrowser

import joblib
import sys

from skimmer import logger
from skimmer.openai_embedding import OpenAIEmbedding
from skimmer.openai_summarizer import OpenAISummarizer
from skimmer.parser import RightBranchingParser, StanzaParser
from skimmer.summary_matching_scorer import Method, SummaryMatchingScorer, \
    SummaryMatchingClauseAbridger, scored_spans_as_html


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
        abridger = SummaryMatchingScorer(parser, embed, summarize)
    elif method == Method.CLAUSE_SUMMARY_MATCHING:
        parser = StanzaParser('en')
        abridger = SummaryMatchingClauseAbridger(parser, embed, summarize)
    else:
        raise Exception(f"invalid method: {method}")

    spans = abridger(doc)
    with tempfile.NamedTemporaryFile('w', suffix='.html', delete=False) as f:
        logger.info(f"Saving HTML output to: {f.name}")
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
    sys.exit(demo())

