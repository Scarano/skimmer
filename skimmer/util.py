from itertools import islice
from typing import Any

from numpy.typing import ArrayLike


def abbrev(s: str, max_len: int=20) -> str:
    if len(s) <= max_len:
        return s
    else:
        return s[:max_len//2] + '\u2026' + s[-max_len//2+1:]


def batched(iterable, n):
    """
    Batch data into lists of length n. The last batch may be shorter.
    batched('ABCDEFG', 3) --> ABC DEF G

    (This implementation taken from https://github.com/python/cpython/issues/98363
    to avoid depending on Python >= 3.12.)
    """
    if n < 1:
        raise ValueError('n must be >= 1')
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

def contiguous_spans(ints: list[int]) -> list[tuple[int, int]]:
    """
    Find contiguous spans of incrementing integers in a list.
    """

    if not ints:
        return []

    spans = []
    start = end = ints[0]

    for i in ints[1:]:
        if i == end + 1:
            end = i
        else:
            spans.append((start, end))
            start = end = i
    spans.append((start, end))

    return spans

def equal_spans(values: ArrayLike) -> list[tuple[int, int, Any]]:
    """
    Find runs of equal values in the input sequence. For each such run, output the start index,
    the end index, and the value
    """

    spans = []
    start = end = 0
    value = values[0]

    for i, v in enumerate(values[1:], 1):
        if v == value:
            end = i
        else:
            spans.append((start, end, value))
            start = end = i
            value = v
    spans.append((start, end, value))

    return spans
