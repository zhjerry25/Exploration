"""Causal, oracle, and sparse-gradient checks for the PKM model.

The loop oracle recomputes product-key addressing per position (score,
top-m, combine, softmax, gather). Sparse-gradient correctness: only the
selected value rows may receive gradient; the sub-key sets must both
receive gradient through the cell softmax.
"""
import math
import unittest

import torch

from st.pkm_model import PKMModel


def model(**overrides):
    args = dict(vocab_size=31, dim=16, heads=2, layers=2, pkm_k=16,
                pkm_m=2, pkm_heads=2)
    args.update(overrides)
    return PKMModel(**args)


def oracle_logits(m, ids):
    """Loop reference for the full forward."""
    batch, n = ids.shape
    x = m.embedding(ids)
    positions = torch.arange(n, device=ids.device)
    for block in m.blocks:
        h = block.attn_norm(x)
        q = block.rope(block.wq(h).view(batch, n, m.heads, block.hd),
                       positions)
        k = block.rope(block.wk(h).view(batch, n, m.heads, block.hd),
                       positions)
        v = block.wv(h).view(batch, n, m.heads, block.hd)
        outs = []
        for t in range(n):
            s = torch.einsum('bhd,bjhd->bhj', q[:, t], k[:, :t + 1])
            alpha = (s / math.sqrt(block.hd)).softmax(-1)
            outs.append(torch.einsum('bhj,bjhd->bhd', alpha, v[:, :t + 1]))
        y = x + block.out(torch.stack(outs, 1).reshape(batch, n, m.dim))
        h2 = block.ffn_norm(y)
        tb = block.table
        qq = tb.wq(h2).view(batch, n, tb.heads, tb.dk)
        q1, q2 = qq.chunk(2, -1)
        s1 = torch.einsum('bnhk,hsk->bnhs', q1, tb.k1)
        s2 = torch.einsum('bnhk,hsk->bnhs', q2, tb.k2)
        v1, i1 = s1.topk(tb.m, dim=-1)
        v2, i2 = s2.topk(tb.m, dim=-1)
        cells = v1[..., None] + v2[..., None, :]
        w = cells.flatten(-2).softmax(-1)
        idx = (i1[..., None] * tb.S + i2[..., None, :]).flatten(-2)
        vals = tb.values[idx]
        read = (w[..., None] * vals).sum(-2)
        x = y + tb.out(read.mean(2))
    return m.lm_head(m.final_norm(x))


class PKMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        torch.manual_seed(123)

    def test_shapes_backward_and_tail_lengths(self):
        for n in [1, 2, 3, 5, 64, 65, 257]:
            for k in [16, 64]:
                with self.subTest(n=n, pkm_k=k):
                    m = model(pkm_k=k)
                    ids = torch.randint(31, (2, n))
                    logits = m(ids, sup=torch.ones(2, n, dtype=torch.bool))
                    self.assertEqual(logits.shape, (2, n, 31))
                    self.assertTrue(torch.isfinite(logits).all())
                    logits.square().mean().backward()
                    for param in m.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                    m.zero_grad(set_to_none=True)

    def test_causality_suffix_and_prefix(self):
        m = model().double().eval()
        ids = torch.randint(31, (2, 33))
        with torch.no_grad():
            full = m(ids)
            for cut in [1, 8, 16, 17, 32]:
                with self.subTest(cut=cut):
                    changed = ids.clone()
                    changed[:, cut:] = (changed[:, cut:] + 7) % 31
                    torch.testing.assert_close(m(changed)[:, :cut],
                                               full[:, :cut],
                                               rtol=1e-9, atol=1e-10)
                    torch.testing.assert_close(m(ids[:, :cut]), full[:, :cut],
                                               rtol=1e-9, atol=1e-10)

    def test_oracle_values_and_gradients(self):
        for extras in [dict(), dict(pkm_k=64), dict(balance=0.1)]:
            with self.subTest(**extras):
                m = model(**extras).double()
                ids = torch.randint(31, (2, 21))
                m.eval()
                with torch.no_grad():
                    torch.testing.assert_close(m(ids), oracle_logits(m, ids),
                                               rtol=1e-9, atol=1e-10)
                m.train()
                probe = torch.randn(2, 21, 31, dtype=torch.double)
                tb = m.blocks[0].table
                params = [m.blocks[0].wq.weight, m.blocks[0].wv.weight,
                          tb.wq.weight, tb.k1, tb.k2, tb.values, tb.out.weight,
                          m.embedding.weight]
                out = m(ids)
                grad_a = torch.autograd.grad((out * probe).sum(), params,
                                             retain_graph=True)
                grad_e = torch.autograd.grad((oracle_logits(m, ids) * probe)
                                             .sum(), params)
                for a, e in zip(grad_a, grad_e):
                    torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)

    def test_sparse_value_gradient(self):
        """Only selected value rows get gradient; sub-keys always do."""
        m = model(pkm_k=256)
        ids = torch.randint(31, (1, 5))
        m(ids).square().mean().backward()
        tb = m.blocks[0].table
        used = set(tb.last_idx.reshape(-1).tolist())
        unused = [r for r in range(tb.rows) if r not in used]
        self.assertTrue(len(unused) > 0, "test needs unused rows")
        g = tb.values.grad
        self.assertEqual(g[unused].abs().max().item(), 0)
        self.assertGreater(g[sorted(used)].abs().sum().item(), 0)
        self.assertGreater(tb.k1.grad.abs().sum().item(), 0)
        self.assertGreater(tb.k2.grad.abs().sum().item(), 0)

    def test_probe_shape_and_range(self):
        m = model()
        ids = torch.randint(31, (2, 17))
        m(ids)
        idx = m.blocks[0].table.last_idx
        self.assertEqual(idx.shape, (2, 17, 2, 4))
        self.assertGreaterEqual(int(idx.min()), 0)
        self.assertLess(int(idx.max()), 16)

    def test_no_gradient_to_future_inputs(self):
        m = model().double()
        ids = torch.randint(31, (1, 35))
        captured = []

        def hook(module, inp, out):
            out.retain_grad()
            captured.append(out)

        h = m.embedding.register_forward_hook(hook)
        logits = m(ids)
        logits[0, 20].square().sum().backward()
        h.remove()
        grad = captured[0].grad
        self.assertEqual(grad[:, 21:].abs().max().item(), 0)
        self.assertGreater(grad[:, :21].abs().sum().item(), 0)

    def test_bf16_autocast_smoke(self):
        m = model().eval()
        ids = torch.randint(31, (2, 33))
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = m(ids).float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_configuration_and_empty_inputs(self):
        for kwargs in [dict(pkm_k=15), dict(pkm_k=16, pkm_m=5),
                       dict(pkm_heads=3), dict(balance=-1.0), dict(layers=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)
        m = model()
        for shape in [(0, 4), (1, 0), (4,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                m(torch.empty(shape, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
