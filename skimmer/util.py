from itertools import islice


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
