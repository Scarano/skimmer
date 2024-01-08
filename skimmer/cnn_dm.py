import itertools
import os
from typing import Optional, Generator, Iterator
import csv

import sys

from skimmer.eval_common import ReferenceSummarySet
from skimmer.util import IndexedEnum


class CNN_DM:
    """
    Loads specified subset of CNN / Daily Mail summarization dataset.
    """

    class DataSplit(IndexedEnum):
        TEST = 'test'
        TRAIN = 'train'
        VALIDATION = 'validation'


    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def read(self, dataSplit: DataSplit, subset: float = 1.0, limit: Optional[int] = None) \
            -> Generator[ReferenceSummarySet, None, None]:

        file = os.path.join(self.data_dir, dataSplit.value + '.csv')

        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            if limit is not None:
                reader: Iterator[dict] = itertools.islice(reader, limit)
            for row in reader:
                article_id = row['id']
                # read last 8 digits of article_id as hexadecimal number and convert to a
                # float between 0 and 1. This number will be uniformly distributed, so we can
                # use it to filter out a deterministically pseudorandom subset of articles
                article_id_float = int(article_id[-8:], 16) / 16**8
                if article_id_float < subset:
                    yield ReferenceSummarySet(row['article'], [row['highlights']])


def demo():
    data_dir = sys.argv[1]
    cnn_dm = CNN_DM(data_dir)
    for ref_summary_set in cnn_dm.read(CNN_DM.DataSplit.TRAIN, limit=10):
        summary = ref_summary_set.summaries[0].replace('\n', ' ')
        print(f"{ref_summary_set.doc[:85]}...")
        print(f"  => {summary[:80]}")
        print()


if __name__ == '__main__':
    demo()