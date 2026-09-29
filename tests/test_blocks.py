"""Chunk-invariance for the streamed encoder blocks: chunk size must never
change values (long-context eval memory depends on this streaming)."""
import unittest

import torch

from st.blocks import HaloMemoryBlock, KVProjection, RotaryEmbedding


class StreamedBlocksTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_halo_chunk_invariance(self):
        dim, heads, b = 16, 2, 8
        rope = RotaryEmbedding(dim // heads)
        ref = HaloMemoryBlock(dim, heads, b, rope, 4, query_chunk_size=64)
        fine = HaloMemoryBlock(dim, heads, b, rope, 4, query_chunk_size=1)
        fine.load_state_dict(ref.state_dict())
        ref, fine = ref.double().eval(), fine.double().eval()
        for n in [1, 3, 7, 8, 9, 37, 64]:
            with self.subTest(n=n):
                x = torch.randn(2, n, dim, dtype=torch.double)
                pos = torch.arange(n)
                with torch.no_grad():
                    torch.testing.assert_close(fine(x, pos), ref(x, pos),
                                               rtol=1e-8, atol=1e-10)

    def test_kv_projection_chunk_invariance(self):
        dim, heads = 16, 2
        rope = RotaryEmbedding(dim // heads)
        ref = KVProjection(dim, heads, rope, use_rope=True)
        fine = KVProjection(dim, heads, rope, use_rope=True, chunk_tokens=4)
        fine.load_state_dict(ref.state_dict())
        ref, fine = ref.double().eval(), fine.double().eval()
        x = torch.randn(2, 37, dim, dtype=torch.double)
        pos = torch.arange(37)
        with torch.no_grad():
            k1, v1 = ref(x, pos)
            k2, v2 = fine(x, pos)
        torch.testing.assert_close(k2, k1, rtol=1e-8, atol=1e-10)
        torch.testing.assert_close(v2, v1, rtol=1e-8, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
