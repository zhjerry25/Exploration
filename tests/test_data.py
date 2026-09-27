"""Checks for the synthetic task batchers, esp. the density-scaled hard MQAR."""
import unittest

import torch

from st import data


class MqarTests(unittest.TestCase):
    def test_deterministic_per_seed(self):
        g1 = torch.Generator().manual_seed(7)
        g2 = torch.Generator().manual_seed(7)
        a = data.mqar_batch(2, 256, g1, "cpu", n_pairs=8, n_queries=2,
                            key_tokens=2)
        b = data.mqar_batch(2, 256, g2, "cpu", n_pairs=8, n_queries=2,
                            key_tokens=2)
        for x, y in zip(a, b):
            if x is None:
                self.assertIsNone(y)
            else:
                self.assertTrue(torch.equal(x, y))

    def test_two_token_keys_unique_and_targets_are_digits(self):
        g = torch.Generator().manual_seed(3)
        n_pairs, n_queries = 32, 4
        idx, tgt, mask, pos = data.mqar_batch(4, 2048, g, "cpu",
                                              n_pairs=n_pairs,
                                              n_queries=n_queries,
                                              key_tokens=2)
        self.assertIsNone(pos)
        self.assertTrue(bool((mask.sum(1) == n_queries).all()))
        self.assertTrue(bool(((tgt[mask] >= 0) & (tgt[mask] <= 9)).all()))
        seq = torch.cat([idx, tgt[:, -1:]], 1)  # [bs, n+1]
        for row in range(seq.shape[0]):
            s = seq[row].tolist()
            keys = set()
            for i in range(len(s) - 2):
                if 34 <= s[i] <= 97 and 34 <= s[i + 1] <= 97 \
                        and 0 <= s[i + 2] <= 9:
                    keys.add((s[i], s[i + 1]))
            self.assertEqual(len(keys), n_pairs)  # no key sampled twice

    def test_query_layout(self):
        """Every masked tgt position is preceded by [Q, k...] in the input."""
        for kt in [1, 2]:
            g = torch.Generator().manual_seed(5)
            idx, tgt, mask, _ = data.mqar_batch(2, 1024, g, "cpu", n_pairs=16,
                                                n_queries=4, key_tokens=kt)
            seq = torch.cat([idx, tgt[:, -1:]], 1)
            for row in range(2):
                for j in mask[row].nonzero().flatten().tolist():
                    with self.subTest(kt=kt, row=row, j=j):
                        self.assertEqual(seq[row, j - kt].item(), data.Q)
                        for u in range(kt):
                            self.assertTrue(34 <= seq[row, j - kt + 1 + u] <= 97)

    def test_resolve_npairs(self):
        self.assertEqual(data.resolve_npairs(4096, 16, 0.03125), 128)
        self.assertEqual(data.resolve_npairs(4096, 16, 0.0), 16)
        self.assertEqual(data.resolve_npairs(16, 16, 0.03125), 1)

    def test_asserts(self):
        g = torch.Generator().manual_seed(0)
        with self.assertRaises(AssertionError):  # sequence too short
            data.mqar_batch(1, 64, g, "cpu", n_pairs=64, n_queries=4,
                            key_tokens=2)
        with self.assertRaises(AssertionError):  # more pairs than keys
            data.mqar_batch(1, 4096, g, "cpu", n_pairs=65, n_queries=1,
                            key_tokens=1)

    def test_passkey_copying_unchanged(self):
        g = torch.Generator().manual_seed(11)
        idx, tgt, mask, pos = data.passkey_batch(2, 128, g, "cpu")
        self.assertEqual(int(mask.sum()), 2 * data.KEY)
        self.assertIsNotNone(pos)
        idx, tgt, mask, pos = data.copying_batch(2, 128, g, "cpu")
        self.assertIsNone(pos)
        self.assertEqual(int(mask.sum()), 2 * (128 // 8))


if __name__ == '__main__':
    unittest.main()
