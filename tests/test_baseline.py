"""Causal and interface checks for the dense transformer baseline."""
import unittest
import copy

import torch

from st.baseline import BaselineModel


def model(**overrides):
    args = dict(vocab_size=31, dim=16, heads=2, layers=2)
    args.update(overrides)
    return BaselineModel(**args)


class BaselineModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_causal_and_prefix_invariance(self):
        m = model().double().eval()
        ids = torch.randint(31, (2, 33))
        with torch.no_grad():
            full = m(ids)
            for cut in [1, 3, 8, 15, 16, 17, 31, 32]:
                with self.subTest(cut=cut):
                    changed = ids.clone()
                    changed[:, cut:] = (changed[:, cut:] + 7) % 31
                    torch.testing.assert_close(m(changed)[:, :cut], full[:, :cut],
                                               rtol=1e-9, atol=1e-10)
                    torch.testing.assert_close(m(ids[:, :cut]), full[:, :cut],
                                               rtol=1e-9, atol=1e-10)

    def test_shapes_sup_zeroing_and_backward(self):
        m = model()
        ids = torch.randint(31, (2, 17))
        sup = torch.zeros(2, 17, dtype=torch.bool)
        sup[:, -3:] = True
        logits = m(ids, sup=sup)
        self.assertEqual(logits.shape, (2, 17, 31))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(logits[:, :-3].abs().max().item(), 0.0)
        logits[sup].square().mean().backward()
        for p in m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_full_block_checkpoint_matches_reference(self):
        torch.manual_seed(9)
        reference = model(backend="math").double()
        checkpointed = copy.deepcopy(reference)
        checkpointed.checkpoint_chunks = True
        for block in checkpointed.blocks:
            block.attention_backend = "math"
        ids = torch.randint(31, (2, 13))
        targets = torch.randint(31, ids.shape)
        sup = torch.zeros_like(ids, dtype=torch.bool)
        sup[:, -4:] = True
        expected = reference(ids, targets=targets, sup=sup)["loss_sum"]
        actual = checkpointed(ids, targets=targets, sup=sup)["loss_sum"]
        torch.testing.assert_close(actual, expected, atol=1.e-10, rtol=1.e-10)
        expected.backward()
        actual.backward()
        for left, right in zip(reference.parameters(), checkpointed.parameters()):
            if left.grad is not None:
                torch.testing.assert_close(left.grad, right.grad, atol=1.e-9, rtol=1.e-8)

    def test_param_count_matches_stack_within_tolerance(self):
        # the E1 pairing: baseline --layers 3 vs stack "Lx2,G"
        from st.stack_model import StackModel
        torch.manual_seed(0)
        base = BaselineModel(256, dim=256, heads=4, layers=3)
        stack = StackModel(256, dim=256, heads=4, block_size=16, topk=64,
                           arch="Lx2,G")
        pb = sum(p.numel() for p in base.parameters())
        ps = sum(p.numel() for p in stack.parameters())
        self.assertLess(abs(pb - ps) / ps, 0.10)

    def test_validation(self):
        for kwargs in [dict(heads=3), dict(layers=0), dict(dim=0),
                       dict(ffn_ratio=0), dict(vocab_size=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)
        with self.assertRaises(ValueError):
            model()(torch.empty(4, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
