import copy
import tempfile
import unittest

import torch

from st.inference import InferenceSession
from st.parallel import ParallelContext
from st.stack_model import StackModel
from st.token_data import TokenDataset


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(617)

    def make_model(self, **kw):
        args = dict(vocab_size=31, dim=16, heads=2, block_size=4, topk=2,
                    arch="Lx2,(G)x2", backend="torch", pos_chunk=3, encoder_chunk=8)
        args.update(kw)
        return StackModel(**args).double()

    def test_chunked_loss_and_parameter_gradients(self):
        ref = self.make_model(topk=64)
        actual = copy.deepcopy(ref)
        actual.checkpoint_chunks = True
        for layer in actual.local:
            layer.checkpoint_chunks = True
        ids = torch.randint(31, (2, 21))
        targets = torch.randint(31, (2, 21))
        mask = torch.rand(2, 21) > .4
        logits = ref(ids)
        expected = torch.nn.functional.cross_entropy(logits[mask].float(), targets[mask], reduction="sum")
        result = actual(ids, targets=targets, sup=mask)
        torch.testing.assert_close(result["loss_sum"], expected, atol=2.e-5, rtol=1.e-6)
        expected.backward()
        result["loss_sum"].backward()
        for (name, rp), (_, ap) in zip(ref.named_parameters(), actual.named_parameters()):
            if rp.grad is not None:
                torch.testing.assert_close(ap.grad, rp.grad, atol=3.e-6, rtol=2.e-5, msg=name)

    def test_all_unsupervised_row_still_has_finite_gradients(self):
        model = self.make_model(checkpoint_chunks=True)
        ids = torch.randint(31, (2, 13))
        targets = torch.full_like(ids, -100)
        targets[1, -2:] = 2
        result = model(ids, targets=targets)
        self.assertEqual(int(result["count"]), 2)
        result["loss_sum"].backward()
        for p in model.parameters():
            if p.grad is not None:
                self.assertTrue(torch.isfinite(p.grad).all())

    def test_streamed_prefill_matches_in_core_and_cleans_disk(self):
        model = self.make_model().eval()
        ids = torch.randint(31, (2, 53))
        pos = torch.tensor([[0, 7, 31, 52], [1, 8, 40, 51]])
        mask = torch.zeros_like(ids, dtype=torch.bool).scatter_(1, pos, True)
        with torch.no_grad():
            expected = model(ids, sup=mask, compact=True)
        for tier in ("cpu", "disk"):
            with tempfile.TemporaryDirectory() as directory:
                with InferenceSession(model, cache=tier, cache_dir=directory,
                        page_tokens=12, encoder_chunk=8, query_chunk=2) as session:
                    session.prefill(ids, pos)
                    actual = torch.cat([l for _, l, _ in session.iter_logits()], 1)
                    torch.testing.assert_close(actual, expected, atol=1.e-10, rtol=1.e-9)
                from pathlib import Path
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_shifted_halo_has_no_long_position_aliasing(self):
        model = self.make_model(arch="L").eval()
        layer = model.local[0]
        x = torch.randn(2, 12, 16, dtype=torch.float64)
        prev = torch.randn(2, 4, 16, dtype=torch.float64)
        with torch.no_grad():
            expected = layer(x, previous=prev, start=4)
            actual = layer(x, previous=prev, start=32*1024*1024)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_sequence_shards_cover_tail_and_preserve_targets(self):
        x = torch.arange(13)[None]
        parts = []
        for rank in range(4):
            context = ParallelContext(cp_size=4, cp_rank=rank)
            local, offset = context.shard(x, 4, -100)
            self.assertEqual(offset, rank*4)
            parts.append(local)
        joined = torch.cat(parts, 1)
        torch.testing.assert_close(joined[:, :13], x)
        self.assertTrue((joined[:, 13:] == -100).all())

    def test_token_file(self):
        with tempfile.TemporaryDirectory() as directory:
            from pathlib import Path
            path = Path(directory)/"tokens.bin"
            path.write_bytes(bytes(range(64)))
            dataset = TokenDataset(path, "uint8")
            ids, target, mask, _ = dataset.batch(2, 17, torch.Generator().manual_seed(0))
            torch.testing.assert_close(ids[:, 1:], target[:, :-1])
            self.assertEqual(tuple(mask.shape), (2, 17))


if __name__ == "__main__":
    unittest.main()
