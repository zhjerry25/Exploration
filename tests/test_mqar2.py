"""Validity checks for the two-hop MQAR generator: the chain a->b->c must be
decodable from the body, tail answers must match it, and no answer may leak
into the visible tail (queries are a permutation)."""
import unittest

import torch

from st import data


def decode_chains(seq):
    """One sequence [n+1] -> (a->b map, b->c map) from token ranges."""
    a2b, b2c = {}, {}
    for j in range(len(seq) - 1):
        x, y = int(seq[j]), int(seq[j + 1])
        if 34 <= x <= 97 and 98 <= y <= 161:
            a2b[x] = y
        if 98 <= x <= 161 and 0 <= y <= 9:
            b2c[x] = y
    return a2b, b2c


class MQAR2DataTests(unittest.TestCase):
    def test_chain_consistency_and_mask(self):
        bs, n, n_pairs, n_queries = 8, 256, 16, 8
        g = torch.Generator().manual_seed(0)
        idx, tgt, mask, pos = data.mqar2_batch(bs, n, g, torch.device("cpu"),
                                               n_pairs=n_pairs,
                                               n_queries=n_queries)
        self.assertIsNone(pos)
        tail = 3 * n_queries
        t = n + 1 - tail
        for row in range(bs):
            full = torch.cat([idx[row], tgt[row, -1:]])  # reconstruct seq
            a2b, b2c = decode_chains(full.tolist())
            self.assertEqual(len(a2b), n_pairs)
            self.assertEqual(len(b2c), n_pairs)
            for q in range(n_queries):
                self.assertEqual(int(full[t + 3 * q]), data.Q)
                a, c = int(full[t + 1 + 3 * q]), int(full[t + 2 + 3 * q])
                self.assertEqual(b2c[a2b[a]], c)  # the chain must decode
            # mask marks exactly the c positions in tgt coordinates
            want = torch.zeros(n, dtype=torch.bool)
            want[t + 1::3] = True
            self.assertTrue(torch.equal(mask[row], want))
            self.assertEqual(int(mask[row].sum()), n_queries)

    def test_queries_never_repeat_within_sequence(self):
        g = torch.Generator().manual_seed(1)
        idx, tgt, mask, _ = data.mqar2_batch(4, 256, g, torch.device("cpu"),
                                             n_pairs=16, n_queries=8)
        t = 256 + 1 - 24
        for row in range(4):
            full = torch.cat([idx[row], tgt[row, -1:]])
            qs = [int(full[t + 1 + 3 * q]) for q in range(8)]
            self.assertEqual(len(set(qs)), 8)

    def test_determinism_and_token_ranges(self):
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        b1 = data.mqar2_batch(2, 128, g1, torch.device("cpu"))
        b2 = data.mqar2_batch(2, 128, g2, torch.device("cpu"))
        self.assertTrue(torch.equal(b1[0], b2[0]))
        self.assertTrue(torch.equal(b1[1], b2[1]))
        self.assertLess(int(b1[0].max()), data.MQAR2_VOCAB)
        self.assertGreaterEqual(int(b1[0].min()), 0)


if __name__ == '__main__':
    unittest.main()
