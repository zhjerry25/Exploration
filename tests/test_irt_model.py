"""Causal, oracle, and degenerate-init checks for the IRT model.

The loop oracle reimplements iterative readout per position per round with
Python loops. read_rounds=1 must coincide with plain causal attention; a
freshly initialized read_rounds=2 model (U_r zero-init) must produce two
identical round outputs, i.e. it degenerates to the T=1 computation with
W_o given by the sum of its two halves.
"""
import math
import unittest

import torch

from st.irt_model import IRTModel


def model(**overrides):
    args = dict(vocab_size=31, dim=16, heads=2, layers=2, read_rounds=2)
    args.update(overrides)
    return IRTModel(**args)


def loop_read(q, k, v, hd):
    """[B,N,H,hd] -> causal attention read [B,N,H,hd], Python loop."""
    n = q.shape[1]
    outs = []
    for t in range(n):
        s = torch.einsum('bhd,bjhd->bhj', q[:, t], k[:, :t + 1])
        alpha = (s / math.sqrt(hd)).softmax(-1)
        outs.append(torch.einsum('bhj,bjhd->bhd', alpha, v[:, :t + 1]))
    return torch.stack(outs, 1)


def oracle_logits(m, ids):
    """Loop reference for the full forward (all positions, layers, rounds)."""
    batch, n = ids.shape
    x = m.embedding(ids)
    positions = torch.arange(n, device=ids.device)
    for block in m.blocks:
        h = block.attn_norm(x)
        q0 = block.rope(block.wq(h).view(batch, n, m.heads, block.hd),
                        positions)
        k = block.rope(block.wk(h).view(batch, n, m.heads, block.hd),
                       positions)
        v = block.wv(h).view(batch, n, m.heads, block.hd)
        m0 = loop_read(q0, k, v, block.hd)
        mcur = m0
        for u in block.u:
            q = q0 + u(mcur.reshape(batch, n, m.dim)).view(
                batch, n, m.heads, block.hd)
            mcur = loop_read(q, k, v, block.hd)
        mout = (torch.cat([m0.reshape(batch, n, m.dim),
                           mcur.reshape(batch, n, m.dim)], -1)
                if block.u else mcur.reshape(batch, n, m.dim))
        y = x + block.out(mout)
        x = y + block.ffn(block.ffn_norm(y))
    return m.lm_head(m.final_norm(x))


class IRTModelTests(unittest.TestCase):
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
            for rounds in [1, 2, 3]:
                with self.subTest(n=n, rounds=rounds):
                    m = model(read_rounds=rounds)
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
        """Suffix perturbation and prefix truncation leave earlier logits
        bit-identical, for every round count."""
        for rounds in [1, 2, 3]:
            m = model(read_rounds=rounds).double().eval()
            ids = torch.randint(31, (2, 33))
            with torch.no_grad():
                full = m(ids)
                for cut in [1, 8, 16, 17, 32]:
                    with self.subTest(rounds=rounds, cut=cut):
                        changed = ids.clone()
                        changed[:, cut:] = (changed[:, cut:] + 7) % 31
                        torch.testing.assert_close(m(changed)[:, :cut],
                                                   full[:, :cut],
                                                   rtol=1e-9, atol=1e-10)
                        torch.testing.assert_close(m(ids[:, :cut]),
                                                   full[:, :cut],
                                                   rtol=1e-9, atol=1e-10)

    def test_oracle_values_and_gradients(self):
        for rounds in [1, 2, 3]:
            with self.subTest(rounds=rounds):
                m = model(read_rounds=rounds).double()
                ids = torch.randint(31, (2, 21))
                m.eval()
                with torch.no_grad():
                    torch.testing.assert_close(m(ids), oracle_logits(m, ids),
                                               rtol=1e-9, atol=1e-10)
                m.train()
                probe = torch.randn(2, 21, 31, dtype=torch.double)
                params = [m.blocks[0].wq.weight, m.blocks[0].wv.weight,
                          m.blocks[0].out.weight, m.blocks[0].ffn[0].weight,
                          m.embedding.weight]
                params += [u.weight for u in m.blocks[0].u]
                out = m(ids)
                grad_a = torch.autograd.grad((out * probe).sum(), params,
                                             retain_graph=True)
                grad_e = torch.autograd.grad((oracle_logits(m, ids) * probe)
                                             .sum(), params)
                for a, e in zip(grad_a, grad_e):
                    torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)

    def test_degenerate_rounds_start_identical(self):
        """Fresh T=2 (U zero-init): both rounds read identically, so the
        block equals a T=1 block whose out projection is the sum of the two
        halves of W_o."""
        m = model(read_rounds=2).double().eval()
        self.assertEqual(len(m.blocks[0].u), 1)
        ids = torch.randint(31, (2, 19))
        batch, n = ids.shape
        with torch.no_grad():
            x = m.embedding(ids)
            positions = torch.arange(n)
            for block in m.blocks:
                h = block.attn_norm(x)
                q0 = block.rope(block.wq(h).view(batch, n, m.heads, block.hd),
                                positions)
                k = block.rope(block.wk(h).view(batch, n, m.heads, block.hd),
                               positions)
                v = block.wv(h).view(batch, n, m.heads, block.hd)
                m0 = loop_read(q0, k, v, block.hd)
                w_l, w_r = block.out.weight.chunk(2, dim=1)  # [dim, H*hd] x2
                y = x + m0.reshape(batch, n, -1) @ (w_l + w_r).T
                x = y + block.ffn(block.ffn_norm(y))
            expected = m.lm_head(m.final_norm(x))
            torch.testing.assert_close(m(ids), expected, rtol=1e-9,
                                       atol=1e-10)

    def test_zero_init_u_still_receives_gradient(self):
        """Zero-init must not be a dead init: U_r's gradient flows through
        the final round's read (W_o's second half is dense)."""
        m = model(read_rounds=2)
        ids = torch.randint(31, (2, 17))
        m(ids).square().mean().backward()
        g = m.blocks[0].u[0].weight.grad
        self.assertIsNotNone(g)
        self.assertGreater(g.abs().sum().item(), 0)

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
        for kwargs in [dict(heads=3), dict(layers=0), dict(read_rounds=0),
                       dict(ffn_ratio=0)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)
        m = model()
        for shape in [(0, 4), (1, 0), (4,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                m(torch.empty(shape, dtype=torch.long))


if __name__ == '__main__':
    unittest.main()
