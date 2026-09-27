"""Independent causal and loop-oracle checks for the stack model.

The end-to-end oracle reimplements the plumbing (block scoring, visibility,
topk, gate, candidate gathering) with Python loops while calling the trusted
modules from st.blocks, and supports any arch spec (multiple/cycled G rounds).
"""
import math
import unittest

import torch
from torch.nn import functional as F

from st.stack_model import StackModel
from st.blocks import _gather_heads


def model(**overrides):
    args = dict(vocab_size=31, dim=16, heads=2, block_size=2, topk=3,
                local_layers=2, pos_chunk=3)
    args.update(overrides)
    return StackModel(**args)


def oracle_logits(m, ids, positions):
    """Loop reference for the full forward at the given supervised positions."""
    b = m.block_size
    batch, n = ids.shape
    x, raw_k, raw_v = m.encode(ids)
    groups = (n + b - 1) // b
    pad = groups * b - n
    kb = F.pad(raw_k, (0, 0, 0, 0, 0, pad)).reshape(batch, groups, b,
                                                    m.heads, m.hd)
    outs = []
    for t in positions:
        k = t // b
        lo = max((k - 1) * b, 0)
        cover_end = (torch.arange(groups) + 1) * b
        vis = cover_end <= (k - 1) * b             # [G]
        z = x[:, t]                                 # [B,dim]
        for rd in m.reads:
            q = rd.wq(rd.q_norm(z)).reshape(batch, m.heads, m.hd)  # [B,H,hd]
            # push: exact block partition function
            ts = torch.einsum('bhd,bgwhd->bhgw', q, kb) / math.sqrt(m.hd)
            s = ts.logsumexp(-1)                         # [B,H,G]
            sm = s.masked_fill(~vis[None, None, :], float('-inf'))
            with torch.no_grad():
                keep = min(m.topk, groups)
                top, sel = sm.topk(keep, dim=-1)
                sel_ok = torch.isfinite(top)
            gate = torch.nan_to_num(sm - sm.logsumexp(-1, keepdim=True),
                                    nan=0.0, neginf=0.0)
            gate_sel = torch.where(sel_ok, torch.gather(gate, 2, sel),
                                   torch.zeros((), dtype=gate.dtype))
            tok = sel[..., None] * b + torch.arange(b)
            tok = tok.reshape(batch, m.heads, -1)
            tok_ok = (sel_ok[..., None].expand(-1, -1, -1, b)
                      .reshape(batch, m.heads, -1)
                      & (tok < n) & (tok <= t))
            kr = _gather_heads(raw_k.transpose(1, 2),
                               tok.clamp(0, n - 1)[:, :, None, :])[:, :, 0]
            vr = _gather_heads(raw_v.transpose(1, 2),
                               tok.clamp(0, n - 1)[:, :, None, :])[:, :, 0]
            bias = gate_sel[..., None].expand(-1, -1, -1, b)
            bias = bias.reshape(batch, m.heads, -1)
            kl, vl = raw_k[:, lo:t + 1], raw_v[:, lo:t + 1]
            s_loc = torch.einsum('bhd,bwhd->bhw', q, kl) / math.sqrt(m.hd)
            s_rem = torch.einsum('bhd,bhkd->bhk', q, kr) / math.sqrt(m.hd) + bias
            alls = torch.cat([s_loc, s_rem], -1)
            mask = torch.cat([torch.ones_like(s_loc, dtype=torch.bool), tok_ok], -1)
            p = alls.masked_fill(~mask, float('-inf')).softmax(-1)
            w = s_loc.shape[-1]
            ctx = torch.einsum('bhw,bwhd->bhd', p[..., :w], vl) \
                + torch.einsum('bhk,bhkd->bhd', p[..., w:], vr)
            z = z + rd.read_out(ctx.reshape(batch, m.dim))
            z = z + rd.ffn(rd.ffn_norm(z))
        outs.append(m.lm_head(m.final_norm(z)))
    return torch.stack(outs, 1)


class StackModelTests(unittest.TestCase):
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
        for n in [1, 3, 4, 5, 16, 17, 65]:
            for use_sup in [False, True]:
                with self.subTest(n=n, use_sup=use_sup):
                    m = model()
                    ids = torch.randint(31, (2, n))
                    sup = None
                    if use_sup:
                        sup = torch.zeros(2, n, dtype=torch.bool)
                        sup[:, -2:] = True
                    logits = m(ids, sup=sup)
                    self.assertEqual(logits.shape, (2, n, 31))
                    self.assertTrue(torch.isfinite(logits).all())
                    if use_sup and n > 2:
                        self.assertEqual(logits[:, :-2].abs().max().item(), 0.0)
                        loss = logits[sup].square().mean()
                    else:
                        loss = logits.square().mean()
                    loss.backward()
                    for param in m.parameters():
                        if param.grad is not None:
                            self.assertTrue(torch.isfinite(param.grad).all())
                    m.zero_grad(set_to_none=True)

    def test_sup_none_equals_all_positions(self):
        m = model().double().eval()
        ids = torch.randint(31, (2, 33))
        with torch.no_grad():
            a = m(ids)
            b_ = m(ids, sup=torch.ones(2, 33, dtype=torch.bool))
        torch.testing.assert_close(a, b_, rtol=1e-10, atol=1e-10)

    def test_end_to_end_oracle_values_and_gradients(self):
        m = model().double()
        ids = torch.randint(31, (2, 21))
        sup = torch.zeros(2, 21, dtype=torch.bool)
        sup[:, -4:] = True
        positions = [17, 18, 19, 20]
        with torch.no_grad():
            m.eval()
            actual = m(ids, sup=sup)
            expected = oracle_logits(m, ids, positions)
            torch.testing.assert_close(actual[:, -4:], expected,
                                       rtol=1e-9, atol=1e-10)
        m.train()
        probe = torch.randn(2, 4, 31, dtype=torch.double)
        params = [m.reads[0].wq.weight, m.reads[0].read_out.weight,
                  m.raw_kv.proj.weight, m.local[0].qkv.weight,
                  m.reads[0].ffn[0].weight, m.embedding.weight]
        out = m(ids, sup=sup)
        grad_a = torch.autograd.grad((out[:, -4:] * probe).sum(), params,
                                     retain_graph=True)
        expected = oracle_logits(m, ids, positions)
        grad_e = torch.autograd.grad((expected * probe).sum(), params)
        for a, e in zip(grad_a, grad_e):
            torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)

    def test_dense_fast_path_matches_sparse_oracle(self):
        """topk >= total blocks triggers the dense fast path; it must agree
        with the sparse reference plumbing (oracle) value-wise, with and
        without sup."""
        m = model(topk=64).double().eval()  # 64 >= 11 blocks -> dense
        ids = torch.randint(31, (2, 21))
        sup = torch.zeros(2, 21, dtype=torch.bool)
        sup[:, -4:] = True
        with torch.no_grad():
            actual = m(ids, sup=sup)
            expected = oracle_logits(m, ids, [17, 18, 19, 20])
            torch.testing.assert_close(actual[:, -4:], expected,
                                       rtol=1e-9, atol=1e-10)
            full = m(ids)
            expected_full = oracle_logits(m, ids, list(range(21)))
            torch.testing.assert_close(full, expected_full, rtol=1e-9, atol=1e-10)
        loss = m(ids, sup=sup).square().mean()
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_grad_ckpt_matches_direct(self):
        """Checkpointed read rounds must produce identical logits and grads."""
        torch.manual_seed(7)
        ref = model(grad_ckpt=False).double()
        ck = model(grad_ckpt=True).double()
        ck.load_state_dict(ref.state_dict())
        ids = torch.randint(31, (2, 21))
        sup = torch.zeros(2, 21, dtype=torch.bool)
        sup[:, -4:] = True
        ref.train(), ck.train()
        out_r, out_c = ref(ids, sup=sup), ck(ids, sup=sup)
        torch.testing.assert_close(out_c, out_r, rtol=1e-9, atol=1e-10)
        probe = torch.randn(2, 21, 31, dtype=torch.double)
        gr = torch.autograd.grad((out_r * probe).sum(), ref.parameters())
        gc = torch.autograd.grad((out_c * probe).sum(), ck.parameters())
        for a, e in zip(gc, gr):
            torch.testing.assert_close(a, e, rtol=1e-8, atol=1e-9)

    def test_shared_weight_cycling(self):
        shared = model(arch="L,(G)x2")
        self.assertIs(shared.reads[0], shared.reads[1])
        indep = model(arch="L,Gx2")
        self.assertIsNot(indep.reads[0], indep.reads[1])
        p_shared = sum(p.numel() for p in shared.parameters())
        p_indep = sum(p.numel() for p in indep.parameters())
        self.assertLess(p_shared, p_indep)
        # cycled model must match the multi-round oracle
        m = model(arch="L,(G)x2", topk=3).double().eval()
        ids = torch.randint(31, (2, 21))
        sup = torch.zeros(2, 21, dtype=torch.bool)
        sup[:, -4:] = True
        with torch.no_grad():
            actual = m(ids, sup=sup)
            expected = oracle_logits(m, ids, [17, 18, 19, 20])
            torch.testing.assert_close(actual[:, -4:], expected,
                                       rtol=1e-9, atol=1e-10)

    def test_arch_parser(self):
        for bad in ["", "G,L", "Lx0", "X", "(L)x0", "(G)", "L,", "Lx2,(G)x"]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                model(arch=bad)
        for good, passes in [("L", 1), ("Lx2,G", 3), ("(L)x3,(G)x2", 5),
                             ("Gx2", 2), ("L,Gx2,Lx1", None)]:
            with self.subTest(good=good):
                if passes is None:
                    with self.assertRaises(ValueError):
                        model(arch=good)
                else:
                    m = model(arch=good)
                    self.assertEqual(len(m.local) + len(m.reads), passes)

    def test_future_perturbation_and_prefix_invariance(self):
        for arch in ["Lx2,G", "Lx2,(G)x2", "Lx2,Gx2"]:
            with self.subTest(arch=arch):
                m = model(arch=arch).double().eval()
                ids = torch.randint(31, (2, 33))
                with torch.no_grad():
                    full = m(ids)
                    for cut in [1, 3, 4, 5, 8, 15, 16, 17, 31, 32]:
                        changed = ids.clone()
                        changed[:, cut:] = (changed[:, cut:] + 7) % 31
                        torch.testing.assert_close(m(changed)[:, :cut],
                                                   full[:, :cut],
                                                   rtol=1e-9, atol=1e-10)
                        torch.testing.assert_close(m(ids[:, :cut])[:, :cut],
                                                   full[:, :cut],
                                                   rtol=1e-9, atol=1e-10)

    def test_no_gradient_to_future_inputs(self):
        m = model().double()
        ids = torch.randint(31, (1, 35))
        sup = torch.zeros(1, 35, dtype=torch.bool)
        sup[0, 20] = True
        captured = []

        def hook(module, inp, out):
            out.retain_grad()
            captured.append(out)

        h = m.embedding.register_forward_hook(hook)
        logits = m(ids, sup=sup)
        logits[0, 20].square().sum().backward()
        h.remove()
        grad = captured[0].grad
        self.assertEqual(grad[:, 21:].abs().max().item(), 0)
        self.assertGreater(grad[:, :21].abs().sum().item(), 0)

    def test_gradient_highway_reaches_unselected_visible_blocks(self):
        m = model(block_size=4, topk=2)
        m.train()
        ids = torch.randint(31, (2, 65))
        sup = torch.zeros(2, 65, dtype=torch.bool)
        sup[:, -6:] = True
        x, raw_k, raw_v = m.encode(ids)
        n = 65
        b = m.block_size
        groups = (n + b - 1) // b
        pos = torch.arange(60, 65)[None].expand(2, 5)
        block_of = pos // b
        cover_end = (torch.arange(groups) + 1) * b
        indexed_end = (block_of - 1) * b
        visible = cover_end[None, None, :] <= indexed_end[:, :, None]
        kb = F.pad(raw_k, (0, 0, 0, 0, 0, groups * b - n)).reshape(
            2, groups, b, m.heads, m.hd)
        rows = torch.arange(2)[:, None]
        q = m.reads[0].wq(m.reads[0].q_norm(x[rows, pos])).reshape(
            2, 5, m.heads, m.hd)
        s = m.push_scores(q, kb, visible)
        s.retain_grad()
        loss = s.logsumexp(-1).square().mean()  # stand-in touching the gate
        gate = torch.nan_to_num(s - s.logsumexp(-1, keepdim=True),
                                nan=0.0, neginf=0.0)
        loss = loss + gate.sum()
        loss.backward()
        sg = s.grad[0, -1]  # [H,G] at last supervised position t=64 (block 16)
        vis_blocks = torch.arange(groups) <= 14
        with torch.no_grad():
            keep = min(m.topk, groups)
            _, sel = s[0, -1].topk(keep, dim=-1)
        chosen = set(sel.flatten().tolist())
        unselected = [j for j in torch.where(vis_blocks)[0].tolist()
                      if j not in chosen]
        self.assertTrue(len(unselected) > 0)
        self.assertGreater(sg[:, unselected].abs().sum().item(), 0)
        self.assertEqual(sg[:, 15:].abs().sum().item(), 0.0)
        for name in ["reads.0.wq.weight", "raw_kv.proj.weight"]:
            p = dict(m.named_parameters())[name]
            self.assertIsNotNone(p.grad, name)
            self.assertGreater(p.grad.norm().item(), 0, name)

    def test_push_scores_matches_manual(self):
        m = model().double()
        ids = torch.randint(31, (2, 33))
        x, raw_k, _ = m.encode(ids)
        n = 33
        b = m.block_size
        groups = (n + b - 1) // b
        kb = F.pad(raw_k, (0, 0, 0, 0, 0, groups * b - n)).reshape(
            2, groups, b, m.heads, m.hd)
        pos = torch.tensor([[7, 20, 32], [7, 20, 32]])
        block_of = pos // b
        cover_end = (torch.arange(groups) + 1) * b
        visible = cover_end[None, None, :] <= ((block_of - 1) * b)[:, :, None]
        q = m.reads[0].wq(m.reads[0].q_norm(x[torch.arange(2)[:, None],
                                              pos])).reshape(2, 3, m.heads,
                                                             m.hd)
        s = m.push_scores(q, kb, visible)
        # manual: per (row, pos, head, block)
        for row in range(2):
            for pi, t in enumerate([7, 20, 32]):
                k = t // b
                for h_ in range(m.heads):
                    for g_ in range(groups):
                        if (g_ + 1) * b <= (k - 1) * b:
                            toks = raw_k[row, g_ * b:(g_ + 1) * b, h_]
                            ref = (q[row, pi, h_] @ toks.T
                                   / math.sqrt(m.hd)).logsumexp(-1)
                            torch.testing.assert_close(
                                s[row, pi, h_, g_], ref, rtol=1e-10, atol=1e-10)
                        else:
                            self.assertEqual(s[row, pi, h_, g_].item(),
                                             float('-inf'))

    def test_pos_chunk_invariance_and_bf16(self):
        m = model().eval()
        other = model(pos_chunk=1).eval()
        other.load_state_dict(m.state_dict())
        ids = torch.randint(31, (2, 33))
        sup = torch.zeros(2, 33, dtype=torch.bool)
        sup[:, -7:] = True
        with torch.no_grad():
            torch.testing.assert_close(m(ids, sup=sup), other(ids, sup=sup),
                                       rtol=1e-5, atol=1e-6)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = m(ids).float().square().mean()
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for p in m.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_diagnose_smoke(self):
        m = model(topk=2).eval()
        ids = torch.randint(31, (2, 33))
        with torch.no_grad():
            rec = m.diagnose(ids)
        self.assertIn("sel_cov", rec)
        self.assertGreaterEqual(rec["sel_cov"], 0.0)
        self.assertLessEqual(rec["sel_cov"], 1.0)
        m_dense = model(topk=64).eval()  # all blocks selected -> coverage 1
        with torch.no_grad():
            rec_dense = m_dense.diagnose(ids)
        self.assertAlmostEqual(rec_dense["sel_cov"], 1.0, places=3)
        self.assertNotIn("gate_margin", rec_dense)  # nothing unselected

    def test_configuration_and_empty_inputs(self):
        for kwargs in [dict(block_size=1), dict(heads=3), dict(topk=0),
                       dict(local_layers=0), dict(ffn_ratio=0),
                       dict(pos_chunk=-1)]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model(**kwargs)
        m = model()
        for shape in [(0, 4), (1, 0), (4,)]:
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                m(torch.empty(shape, dtype=torch.long))
        with self.assertRaises(ValueError):
            ids = torch.randint(31, (2, 8))
            ragged = torch.zeros(2, 8, dtype=torch.bool)
            ragged[0, -3:] = True
            ragged[1, -2:] = True
            m(ids, sup=ragged)


if __name__ == '__main__':
    unittest.main()
