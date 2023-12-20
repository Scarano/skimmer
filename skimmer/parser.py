import logging
from abc import ABC
from dataclasses import dataclass
from pprint import pprint
from typing import Iterable, TypeVar, Optional

import joblib
import stanza

from skimmer.util import contiguous_spans


def reverse_dict(d):
    reversed_dict = {}
    for key, value in d.items():
        reversed_dict.setdefault(value, []).append(key)
    return reversed_dict


T = TypeVar('T')

def dfs(node: T, visited: set[T], graph: dict[T, set[T]]):
    visited.add(node)
    for neighbor in graph.get(node, []):
        if neighbor not in visited:
            dfs(neighbor, visited, graph)

def transitive_closure(d: dict[T, set[T]], keys: Iterable[T]) -> dict[T, set[T]]:
    """
    :param d: Dict defining relation defining the transitive closure. (a, b) is in the relation
        iff `b in d[a]`.
    :param keys: Keys desired in result
    :return: Dict mapping each key in keys to the transitive closure of the key
    """

    closure = {}
    for key in keys:
        visited = {key}
        dfs(key, visited, d)
        closure[key] = visited
    return closure


@dataclass
class DepParse:
    text: str

    # Word strings.
    # Note that these may not match original text due to Stanza's multi-word tokenization.
    words: list[str]

    # Start and end character offsets for each word in `words`.
    spans: list[tuple[int, int]]

    # Index of head of word corresponding to `words`. Zero-based.
    heads: list[int]

    # Relationship to head (see https://universaldependencies.org/u/dep/)
    relations: list[str]

    # A constituent of word i is considered to be the set of indexes reachable by reverse-head
    # relations from i. This is given as a list of ints, not a range, because in theory, a
    # dependency parse can yield non-contiguous constituents. (Though I'm not sure if Stanza's
    # parser actually does or not.)
    constituents: list[list[int]]

    @property
    def start(self) -> int:
        return self.spans[0][0]

    @property
    def end(self) -> int:
        return self.spans[-1][1]

    def __len__(self) -> int:
        return len(self.words)

    def subspans(self, included_tokens: list[int]) -> list[tuple[int, int]]:
        if len(included_tokens) == 0:
            return []
        elif len(included_tokens) == 1:
            return [self.spans[included_tokens[0]]]
        else:
            return [(self.spans[i][0], self.spans[j][1])
                    for i, j in contiguous_spans(included_tokens)]


class Parser(ABC):
    def parse(self, text: str) -> list[DepParse]:
        raise NotImplementedError


class StanzaParser(Parser):

    def __init__(self, language: str, memory: Optional[joblib.Memory] = None):
        # This includes mwt (multi-word tokenization) because some languages require it.
        # For English, though, it just results in a warning.
        self.pipeline = stanza.Pipeline(language,
                                        processors={'tokenize': 'spacy',
                                                    # 'mwt': 'default',
                                                    'pos': 'default',
                                                    'lemma': 'default',
                                                    'depparse': 'default'},
                                        download_method=stanza.DownloadMethod.REUSE_RESOURCES)

        self.parse_func = lambda text: StanzaParser.uncached_parse(self.pipeline, text)
        if memory:
            self.parse_func = memory.cache(self.parse_func)


    @staticmethod
    def uncached_parse(pipeline: stanza.Pipeline, text: str) -> list[DepParse]:

        parsed_doc = pipeline(text)

        parses = []
        for sentence in parsed_doc.sentences:
            if not sentence.words:
                continue

            if not all(w.id == i + 1 for i, w in enumerate(sentence.words)):
                logging.warning(f"Word ID sanity check failed ('{sentence.text}")

            # Build map from word indexes to their heads' indexes
            # Stanza's word indexes are off by one to accommodate the implicit initial ROOT node,
            # so we'll subtract 1 to that they line up with the actual word indexes.
            # Beware: the ROOT head will now be -1 instead of 0.
            head_map = {w.id - 1: w.head - 1 for w in sentence.words}

            # Reverse it -- build map from word index to list of word indexes whose heads are that word
            rev_map = reverse_dict(head_map)

            # Now build map from word index to set of all indexes dominated by that word
            constituent_map = transitive_closure(rev_map, head_map.keys())

            # print(sentence.words[0].parent.start_char)

            parses.append(DepParse(
                sentence.text,
                [w.text for w in sentence.words],
                [(w.parent.start_char, w.parent.end_char) for w in sentence.words],
                [head_map[i] for i in range(len(sentence.words))],
                [w.deprel for w in sentence.words],
                [sorted(constituent_map[i]) for i in range(len(sentence.words))]
            ))

        return parses

    def parse(self, text: str) -> list[DepParse]:
        return self.parse_func(text)



class RightBranchingParser(Parser):
    """
    A degenrate parser that creates a right-branching dependency parse. (But it's still does
    useful tokenization and sentence-segmentation.)

    This was created as a quick replacement for Parser for cases where I don't actually need
    real parsing (which turns out to be less useful for this problem than I originally hoped).
    This way, I don't have to re-write code that uses the DepParse object.

    (If this were production code, a better refactor would be called for.)
    """

    def __init__(self, language: str):
        self.pipeline = stanza.Pipeline(language,
                                        package=None,
                                        processors={'tokenize': 'spacy'},
                                        download_method=stanza.DownloadMethod.REUSE_RESOURCES)

    def parse(self, text: str) -> list[DepParse]:
        doc = self.pipeline(text)
        parses = []
        for sentence in doc.sentences:
            parses.append(DepParse(
                sentence.text,
                [w.text for w in sentence.words],
                [(w.parent.start_char, w.parent.end_char) for w in sentence.words],
                [i - 1 for i in range(len(sentence.words))],
                ['unknown' for _ in sentence.words],
                [list(range(i, len(sentence.words))) for i in range(len(sentence.words))]
            ))
        return parses


def demo():
    text = "When he first met the FTX founder Sam Bankman-Fried in late 2021, he took the " \
           "cargo-shorted chief executive on a walk through the eucalyptus trees near his " \
           "Berkeley, Calif., home."
    # text = (
    #     "Moral philosophy, or the science of human nature, may be treated after two different "
    #     "manners; each of which has its peculiar merit, and may contribute to the "
    #     "entertainment, instruction, and reformation of mankind. The one considers man chiefly "
    #     "as born for action; and as influenced in his measures by taste and sentiment; pursuing "
    #     "one object, and avoiding another, according to the value which these objects seem to "
    #     "possess, and according to the light in which they present themselves. As virtue, of all "
    #     "objects, is allowed to be the most valuable, this species of philosophers paint her in "
    #     "the most amiable colours; borrowing all helps from poetry and eloquence, and treating "
    #     "their subject in an easy and obvious manner, and such as is best fitted to please the "
    #     "imagination, and engage the affections. They select the most striking observations and "
    #     "instances from common life; place opposite characters in a proper contrast; and alluring "
    #     "us into the paths of virtue by the views of glory and happiness, direct our steps in "
    #     "these paths by the soundest precepts and most illustrious examples. They make us feel "
    #     "the difference between vice and virtue; they excite and regulate our sentiments; and so "
    #     "they can but bend our hearts to the love of probity and true honour, they think, that "
    #     "they have fully attained the end of all their labours.")
    # text = (
    #     "My third maxim was to endeavor always to conquer myself rather than fortune, and change "
    #     "my desires rather than the order of the world, and in general, accustom myself to the "
    #     "persuasion that, except our own thoughts, there is nothing absolutely in our power; so "
    #     "that when we have done our best in things external to us, all wherein we fail of success "
    #     "is to be held, as regards us, absolutely impossible: and this single principle seemed to "
    #     "me sufficient to prevent me from desiring for the future anything which I could not "
    #     "obtain, and thus render me contented; for since our will naturally seeks those objects "
    #     "alone which the understanding represents as in some way possible of attainment, it is "
    #     "plain, that if we consider all external goods as equally beyond our power, we shall no "
    #     "more regret the absence of such goods as seem due to our birth, when deprived of them "
    #     "without any fault of ours, than our not possessing the kingdoms of China or Mexico, and "
    #     "thus making, so to speak, a virtue of necessity, we shall no more desire health in "
    #     "disease, or freedom in imprisonment, than we now do bodies incorruptible as diamonds, or "
    #     "the wings of birds to fly with."
    # )

    parser = StanzaParser('en')

    parses = parser.parse(text)

    for parse in parses:
        for i, w in enumerate(parse.words):
            start = parse.spans[parse.constituents[i][0]][0]
            end = parse.spans[parse.constituents[i][-1]][1]
            cons_text = text[start:end]
            print(f"{i:3} {w:10}: {parse.relations[i]:10} {start}->{end} {cons_text}")


if __name__ == '__main__':
    demo()

