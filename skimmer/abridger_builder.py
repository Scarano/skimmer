import os

import joblib

from skimmer import embedding_scorer
from skimmer import summary_matching_scorer
from skimmer.abridger import Abridger
from skimmer.embedding_scorer import EmbeddingScorer
from skimmer.summary_matching_scorer import SummaryMatchingScorer, SummaryMatchingClauseScorer
from skimmer.parser import RightBranchingParser, StanzaParser


CONFIG_METHOD = 'method'

CONFIG_EMBEDDING = 'embedding'
CONFIG_EMBEDDING_OPENAI = 'openai'

CONFIG_SUMMARY = 'summary'
CONFIG_SUMMARY_OPENAI = 'openai'

CONFIG_CHUNK_SIZE = 'chunk-size'
CONFIG_CHUNK_SIZE_DEFAULT = 30

CONFIG_LENGTH_PENALTY = 'length-penalty'
CONFIG_LENGTH_PENALTY_DEFAULT = 1.0

CONFIG_ABRIDGE_THRESHOLD = 'abridge-threshold'
CONFIG_ABRIDGE_THRESHOLD_DEFAULT = 0.5


class InvalidConfigException(Exception):
    def __init__(self, config, key):
        super().__init__(f"Invalid configuration {key}: {config.get(key, '(unspecified)')}")


def build_scorer_from_config(config: dict, work_dir: str):

    embed_memory = joblib.Memory(os.path.join(work_dir, 'embedding_cache'), mmap_mode='c',
                                 verbose=0)

    if config.get(CONFIG_EMBEDDING) == CONFIG_EMBEDDING_OPENAI:
        from skimmer.openai_embedding import OpenAIEmbedding
        embed = OpenAIEmbedding(memory=embed_memory)
    else:
        raise InvalidConfigException(config, CONFIG_EMBEDDING)

    if config.get(CONFIG_SUMMARY, '') == '':
        summarize = None
    else:
        summary_cache_path = os.path.join(work_dir, f'summary_cache_{config[CONFIG_SUMMARY]}')
        summary_memory = joblib.Memory(summary_cache_path, mmap_mode='c', verbose=0)
        if config[CONFIG_SUMMARY] == CONFIG_SUMMARY_OPENAI:
            from skimmer.openai_summarizer import OpenAISummarizer
            summarize = OpenAISummarizer(memory=summary_memory)
        else:
            raise InvalidConfigException(config, CONFIG_SUMMARY)

    method_str = config[CONFIG_METHOD]

    if embedding_scorer.Method.contains_value(method_str):
        method = embedding_scorer.Method.of(method_str)

        parser = RightBranchingParser('en')

        chunk_size = config.get(CONFIG_CHUNK_SIZE, CONFIG_CHUNK_SIZE_DEFAULT)

        return EmbeddingScorer(method, chunk_size, parser, embed, summarize)

    elif summary_matching_scorer.Method.contains_value(method_str):
        method = summary_matching_scorer.Method.of(method_str)

        match method:
            case summary_matching_scorer.Method.SENTENCE_SUMMARY_MATCHING:

                parser = RightBranchingParser('en')
                return SummaryMatchingScorer(parser, embed, summarize)

            case summary_matching_scorer.Method.CLAUSE_SUMMARY_MATCHING:

                parser = StanzaParser('en')
                length_penalty = config.get(CONFIG_LENGTH_PENALTY, CONFIG_LENGTH_PENALTY_DEFAULT)
                return SummaryMatchingClauseScorer(parser, embed, summarize, length_penalty)

            case _:
                raise Exception("bug")
    else:
        raise InvalidConfigException(config, CONFIG_METHOD)


def build_abridger_from_config(config, work_dir):
    scorer = build_scorer_from_config(config, work_dir)
    threshold = config.get(CONFIG_ABRIDGE_THRESHOLD, CONFIG_ABRIDGE_THRESHOLD_DEFAULT)
    return Abridger(scorer, threshold)
