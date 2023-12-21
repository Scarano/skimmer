import argparse
import os
import tempfile
import webbrowser

import joblib
import sys

from skimmer import logger, openai_summarizer
from skimmer.openai_embedding import OpenAIEmbedding
from skimmer.openai_summarizer import OpenAISummarizer
from skimmer.parser import RightBranchingParser, StanzaParser
from skimmer.summary_matching_scorer import Method, SummaryMatchingScorer, \
    SummaryMatchingClauseScorer, scored_spans_as_html


def score_to_html(doc, method, cache_dir, summarize_prompt: str, length_penalty=0.0):
    """
    Do a demo run of the EmbeddingAbridger on a single doc.
    :param doc: Document to abridge
    :param method: Method to use for abridging
    """
    if cache_dir is not None:
        parse_memory = joblib.Memory(os.path.join(cache_dir, 'parse_cache'),
                                     mmap_mode='c', verbose=0)
        embed_memory = joblib.Memory(os.path.join(cache_dir, 'embedding_cache_openai'),
                                     mmap_mode='c', verbose=0)
        summarize_memory = joblib.Memory(os.path.join(cache_dir, 'summary_cache_openai'),
                                         mmap_mode='c', verbose=0)
    else:
        parse_memory = None
        embed_memory = None
        summarize_memory = None
    embed = OpenAIEmbedding(memory=embed_memory)
    summarize = OpenAISummarizer(prompt_name=summarize_prompt, memory=summarize_memory)
    if method == Method.SENTENCE_SUMMARY_MATCHING:
        parser = RightBranchingParser('en')
        scorer = SummaryMatchingScorer(parser, embed, summarize)
    elif method == Method.CLAUSE_SUMMARY_MATCHING:
        parser = StanzaParser('en', parse_memory)
        scorer = SummaryMatchingClauseScorer(parser, embed, summarize,
                                             length_penalty=length_penalty)
    else:
        raise Exception(f"invalid method: {method}")

    spans = scorer(doc)
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
    parser.add_argument('--summarize-prompt', type=str, default='v1',
                        choices=list(openai_summarizer.SUMMARIZE_PROMPTS.keys()))
    parser.add_argument('--length-penalty', type=float, default=0.0)
    parser.add_argument('--cache-dir', type=str)
    args = parser.parse_args()

    method = Method.of(parser.parse_args().method)

    with open(args.doc) as f:
        doc = f.read()
    score_to_html(doc, method, args.cache_dir, args.summarize_prompt,
                  length_penalty=args.length_penalty)

    return 0


if __name__=='__main__':
    sys.exit(demo())

