import unittest
from unittest import TestCase

from skimmer.util import batched_adaptive


class BatchedAdaptiveTests(TestCase):
    def test_batched_adaptive(self):
        self.assertEqual(list(batched_adaptive('ABCDEFG', 3)),
                         [['A', 'B', 'C'], ['D', 'E'], ['F', 'G']])
        self.assertEqual(list(batched_adaptive('ABCDEF', 3)),
                         [['A', 'B', 'C'], ['D', 'E', 'F']])
        self.assertEqual(list(batched_adaptive('ABCDEF', 6)),
                         [['A', 'B', 'C', 'D', 'E', 'F']])
        self.assertEqual(list(batched_adaptive('ABC', 1)),
                         [['A'], ['B'], ['C']])

if __name__ == '__main__':
    unittest.main()
