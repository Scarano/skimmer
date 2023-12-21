from enum import Enum
from itertools import islice
import time
from typing import Any, Iterable, TypeVar, Generator, Callable, Type

from numpy.typing import ArrayLike


def abbrev(s: str, max_len: int=20) -> str:
    if len(s) <= max_len:
        return s
    else:
        return s[:max_len//2] + '\u2026' + s[-max_len//2+1:]


T = TypeVar('T')
def batched(iterable: Iterable[T], n: int) -> Generator[list[T], None, None]:
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


def batched_adaptive(iterable: Iterable[T], n: int) -> Generator[list[T], None, None]:
    """
    Batch data into lists of length n, except the last 2 batches, which are divided evenly.
    That is, if the last batch would have k items (len(iterable) mod n = k), then
    second-to-last batch and last batch will both have (n + k)/2 items. This avoids
    very small batches (with length < n/2).

    batched_adaptive('ABCDEFG', 3) --> ABC DE FG
    """
    for batch, next_batch in successive_pairs(batched(iterable, n)):
        if next_batch is None or len(next_batch) == n:
            yield batch
        else:
            split = (n + len(next_batch)) // 2
            yield batch[:split]
            yield batch[split:] + next_batch
            return

def successive_pairs(iterable: Iterable[T]) -> Iterable[tuple[T, T]]:
    """
    Yields pairs of (item, next_item) from the given iterable.
    The last element is paired with None.
    """
    iterator = iter(iterable)

    item = next(iterator)
    for next_item in iterator:
        yield item, next_item
        item = next_item
    yield item, None


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


class IndexedEnum(Enum):
    @classmethod
    def of(cls, s: str):
        for instance in cls:
            if instance.value == s:
                return instance
        raise Exception(f"Unknown {cls.__name__} '{s}'")

    @classmethod
    def contains_value(cls, s: str) -> bool:
        return any(m.value == s for m in cls)


def with_retry(fn: Callable[[], T], transient_exception: Type[BaseException],
               max_retries: int, pause: float) -> T:
    try:
        return fn()
    except transient_exception as e:
        if max_retries > 0:
            time.sleep(pause)
            return with_retry(fn, transient_exception, max_retries - 1, pause)
        else:
            return e

