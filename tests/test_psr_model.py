"""Causal, oracle, and degenerate-equivalence checks for the PSR model.

The loop oracle reimplements pair-state readout per position with Python
loops; the degenerate configuration (p_mode="ones", write_gelu off,
r_c == d_a) must coincide with plain causal attention bit-for-bit, and a
freshly initialized p_mode="learned" model must coincide with p_mode="ones"
(wp starts as an all-ones projection).
"""
import math
import unittest

import torch
from torch.nn import functional as F

from st.psr_model import PSRModel


def model(**overrides):
    args = dict(vocab_size=31, dim=16, heads=2, layers=2, d_a=8, r_c=16)
    args.update(overrides)
    return PSRModel(**args)


def oracle_logits(m, ids):
    """Loop reference for the full forward (all positions, all layers)."""
    batch, n = ids.shape
    x = m.embedding(ids)
    positions = torch.arange(n, device=ids.device)
    for block in m.blocks:
        h = block.attn_norm(x)
        q = block.wq(h).view(batch, n, m.heads, block.d_a)
        k = block.wk(h).view(batch, n, m.heads, block.d_a)
        c = block.wc(h).view(batch, n, m.heads, block.r_c)
        if block.write_gelu:
            c = F.gelu(c)
        q, k = m.rope(q, positions), m.rope(k, positions)
        outs = []
        for t in range(n):
            s = torch.einsum('bhd,bjhd->bhj', q[:, t], k[:, :t + 1])
            alpha = (s / math.sqrt(block.d_a)).softmax(-1)
            mt = torch.einsum('bhj,bjhd->bhd', alpha, c[:, :t + 1])
            if block.wp is not None:
                pt = block.wp(h[:, t]).view(batch, m.heads, block.r_c)
                mt = pt * mt
            outs.append(mt)
        ctx = torch.stack(outs, 1).reshape(batch, n, m.heads * block.r_c)
        y = x + block.out(ctx)
        x = y + block.ffn(block.ffn_norm(y))
    return m.lm_head(m.final_norm(x))


class PSRModelTests(unittest.TestCase):
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
            for extras in [dict(), dict(p_mode="ones"),
                           dict(write_gelu=False)]:
                with self.subTest(n=n, **extras):
                    m = model(**extras)
                    ids = torch.randint(31, (2, n))
                    sup = torch.ones(2, n, dtype=torch.bool)
                    logits = m(ids, sup=sup)
                    self.assertEqual(logits.shape, (2, n, 31))
                    self.assertTrue(torch.isfinite(logits).all())
                    logits.square().mean().backward()
                    for param in m.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                    m.zero_grad(set_to_none=True)

    def test_sup_is_accepted_and_ignored(self):
        m = model().double().eval()
        ids = torch.randint(31, (2, 17))
        sup = torch.zeros(2, 17, dtype=torch.bool)
        sup[:, -3:] = True
        with torch.no_grad():
            torch.testing.assert_close(m(ids), m(ids, sup=sup),
                                       rtol=0, atol=0)

    def test_causality_suffix_and_prefix(self):
        """Suffix perturbation and prefix truncation leave earlier logits
        bit-identical — the single most important test in this codebase."""
        m = model().double().eval()
        ids = torch.randint(31, (2, 33))
        with torch.no_grad():
            full = m(ids)
            for cut in [1, 3, 4, 5, 8, 15, 16, 17, 31, 32]:
                with self.subTest(cut=cut):
                    changed = ids.clone()
                    changed[:, cut:] = (changed[:, cut:] + 7) % 31
                    torch.testing.assert_close(m(changed)[:, :cut],
                                               full[:, :cut],
                                               rtol=1e-9, atol=1e-10)
                    torch.testing.assert_close(m(ids[:, :cut]), full[:, :cut],
                                               rtol=1e-9, atol=1e-10)

    def test_oracle_values_and_gradients(self):
        for extras in [dict(), dict(p_mode="ones"), dict(write_gelu=False)]:
            with self.subTest(**extras):
                m = model(**extras).double()
                ids = torch.randint(31, (2, 21))
                m.eval()
                with torch.no_grad():
                    actual = m(ids)
                    expected = oracle_logits(m, ids)
                    torch.testing.assert_close(actual, expected,
                                               rtol=1e-9, atol=1e-10)
                m.train()
                probe = torch.randn(2, 21, 31, dtype=torch.double)
                params = [m.blocks[0].wq.weight, m.blocks[0].wc.weight,
                          m.blocks[0].out.weight, m.blocks[0].ffn[0].weight,
                          m.embedding.weight]
                if m.blocks[0].wp is not None:
                    params.append(m.blocks[0].wp.weight)
                    params.append(m.blocks[0].wp.bias)
                out = m(ids)
                grad_a = torch.autograd.grad((out * probe).sum(), params,
                                             retain_graph=True)
                grad_e = torch.autograd.grad((oracle_logits(m, ids) * probe)
                                             .sum(), params)
                for a, e in zip(grad_a, grad_e):
                    torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)

    def test_degenerate_is_standard_attention(self):
        """p=ones, linear c, r_c == d_a: the model must equal a hand-rolled
        plain causal attention reference bit-for-bit."""
        m = model(p_mode="ones", write_gelu=False, r_c=8).double().eval()
        ids = torch.randint(31, (2, 19))
        batch, n = ids.shape
        with torch.no_grad():
            x = m.embedding(ids)
            positions = torch.arange(n)
            for block in m.blocks:
                h = block.attn_norm(x)
                q = block.wq(h).view(batch, n, m.heads, block.d_a)
                k = block.wk(h).view(batch, n, m.heads, block.d_a)
                v = block.wc(h).view(batch, n, m.heads, block.d_a)
                q, k = m.rope(q, positions), m.rope(k, positions)
                ctx = torch.zeros(batch, n, m.heads, block.d_a,
                                  dtype=x.dtype)
                for t in range(n):
                    s = torch.einsum('bhd,bjhd->bhj', q[:, t], k[:, :t + 1])
                    alpha = (s / math.sqrt(block.d_a)).softmax(-1)
                    ctx[:, t] = torch.einsum('bhj,bjhd->bhd', alpha,
                                             v[:, :t + 1])
                y = x + block.out(ctx.reshape(batch, n, -1))
                x = y + block.ffn(block.ffn_norm(y))
            expected = m.lm_head(m.final_norm(x))
            torch.testing.assert_close(m(ids), expected,
                                       rtol=1e-9, atol=1e-10)

    def test_learned_p_starts_at_degenerate(self):
        """A fresh p_mode='learned' model (wp zero-weight, ones-bias) must
        produce exactly the p_mode='ones' outputs at init."""
        base = model(p_mode="ones")
        learned = model(p_mode="learned")
        learned.load_state_dict(base.state_dict(), strict=False)
        ids = torch.randint(31, (2, 23))
        with torch.no_grad():
            torch.testing.assert_close(base(ids), learned(ids),
                                       rtol=0, atol=0)

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
        for kwargs in [dict(heads=3), dict(d_a=7), dict(layers=0),
                       dict(ffn_ratio=0), dict(r_c=0), dict(p_mode="x")]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)
        m = model()
        for shape in [(0, 4), (1, 0), (4,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                m(torch.empty(shape, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
