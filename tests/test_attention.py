"""Independent eager oracle for the exact gated dense operator and backward."""
import math
import unittest

import torch

from st.attention import dense_attention
from st.sparse import TensorPages, sparse_attention


def eager_attention(q, k, v, positions, block_size, topk=None):
    batch, nq, heads, dim = q.shape
    n, b = k.shape[1], block_size
    outputs = []
    for batch_id in range(batch):
        query_outputs = []
        for qi in range(nq):
            p = int(positions[batch_id, qi])
            head_outputs = []
            for h in range(heads):
                if p < 0:
                    head_outputs.append(q[batch_id, qi, h]*0 + k[batch_id, :, h].sum(0)*0 + v[batch_id, :, h].sum(0)*0)
                    continue
                a = k[batch_id, :p+1, h] @ q[batch_id, qi, h] / math.sqrt(dim)
                visible = max(0, p//b-1)
                selected = list(range(visible))
                gates = None
                if visible:
                    s = a[:visible*b].reshape(visible, b).logsumexp(-1)
                    gates = s.log_softmax(0)
                    if topk is not None and topk < visible:
                        selected = s.topk(topk).indices.tolist()
                indices, logits = [], []
                for j in selected:
                    indices.extend(range(j*b, (j+1)*b))
                    logits.append(a[j*b:(j+1)*b]+gates[j])
                lo = visible*b
                indices.extend(range(lo, p+1))
                logits.append(a[lo:p+1])
                probs = torch.cat(logits).softmax(0)
                head_outputs.append(probs @ v[batch_id, indices, h])
            query_outputs.append(torch.stack(head_outputs))
        outputs.append(torch.stack(query_outputs))
    return torch.stack(outputs)


class ExactAttentionTests(unittest.TestCase):
    def test_forward_and_all_input_gradients(self):
        torch.manual_seed(981)
        for n, b in ((1, 2), (7, 3), (17, 4), (33, 8)):
            with self.subTest(n=n, block=b):
                q = torch.randn(2, 5, 2, 6, dtype=torch.float64, requires_grad=True)
                k = torch.randn(2, n, 2, 6, dtype=torch.float64, requires_grad=True)
                v = torch.randn_like(k, requires_grad=True)
                pos = torch.tensor([[0, min(3, n-1), n-1, -1, n//2], [n-1, -1, 0, n//2, n-1]])
                expected = eager_attention(q, k, v, pos, b)
                actual = dense_attention(q, k, v, pos, b, "torch", q_chunk=2, kv_chunk=2*b)
                torch.testing.assert_close(actual, expected, atol=1.e-11, rtol=1.e-10)
                probe = torch.randn_like(actual)
                eg = torch.autograd.grad((expected*probe).sum(), (q, k, v))
                ag = torch.autograd.grad((actual*probe).sum(), (q, k, v))
                for a, e in zip(ag, eg):
                    torch.testing.assert_close(a, e, atol=1.e-10, rtol=1.e-9)

    def test_gate_gradient_finite_difference(self):
        torch.manual_seed(4)
        q = torch.randn(1, 2, 1, 4, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 13, 1, 4, dtype=torch.float64, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        pos = torch.tensor([[9, 12]])
        self.assertTrue(torch.autograd.gradcheck(
            lambda q, k, v: dense_attention(q, k, v, pos, 2, "torch", 2, 6),
            (q, k, v), eps=1.e-6, atol=1.e-5, rtol=1.e-4))

    def test_sparse_pages_and_sparse_dense_degenerate(self):
        torch.manual_seed(124)
        q = torch.randn(2, 4, 3, 8, dtype=torch.float64)
        k = torch.randn(2, 45, 3, 8, dtype=torch.float64)
        v = torch.randn_like(k)
        pos = torch.tensor([[0, 7, 44, 22], [13, 44, 1, 43]])
        for keep in (1, 2, 64):
            expected = eager_attention(q, k, v, pos, 4, keep)
            for page in (4, 12, 64):
                with self.subTest(keep=keep, page=page):
                    actual = sparse_attention(q, TensorPages(k, v, page), pos, 4, keep, "torch", score_page_tokens=8)
                    torch.testing.assert_close(actual, expected, rtol=1.e-10, atol=1.e-11)


if __name__ == "__main__":
    unittest.main()
