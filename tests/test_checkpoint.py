import copy
from pathlib import Path
import tempfile
import unittest

import torch

from st import checkpoint
from st.parallel import ParallelContext
from st.stack_model import StackModel


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_restores_next_optimizer_step_and_data(self):
        torch.manual_seed(38)
        config = {"model": dict(vocab_size=31, dim=16, heads=2, block_size=4,
                                topk=64, arch="(L)x2,(G)x2"), "runtime": {}}
        model = StackModel(**config["model"], backend="torch", encoder_chunk=8)
        opt = torch.optim.AdamW(model.parameters(), lr=.001)
        g = torch.Generator().manual_seed(82)

        def step(m, o, generator):
            ids = torch.randint(31, (2, 17), generator=generator)
            target = torch.randint(31, (2, 17), generator=generator)
            o.zero_grad(set_to_none=True)
            result = m(ids, targets=target)
            loss = result["loss_sum"]/result["count"]
            loss.backward()
            o.step()
            return ids, loss.detach()

        step(model, opt, g)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"nested"/"checkpoint.pt"
            checkpoint.save(path, model, opt, config, 0, g, ParallelContext())
            expected_ids, expected_loss = step(model, opt, g)
            ck = checkpoint.load(path)
            other = StackModel(**checkpoint.model_config(ck), backend="torch", encoder_chunk=8)
            other.load_state_dict(ck["model"])
            other_opt = torch.optim.AdamW(other.parameters(), lr=.001)
            other_opt.load_state_dict(ck["opt"])
            other_g = torch.Generator()
            checkpoint.restore_rng(ck["rng_by_rank"][0], other_g)
            actual_ids, actual_loss = step(other, other_opt, other_g)
            torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
            torch.testing.assert_close(actual_loss, expected_loss, atol=0, rtol=0)
            for p, q in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)
            self.assertEqual(ck["model"]["embedding.weight"].data_ptr(), ck["model"]["lm_head.weight"].data_ptr())
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_legacy_model_structure(self):
        config = checkpoint.model_config({"args": {"d": 32, "b": 8, "heads": 4,
                    "arch": "L,G", "task": "lm", "read_m": 4, "model": "stack"}})
        self.assertEqual(config["vocab_size"], 256)
        self.assertEqual(config["dim"], 32)
        self.assertEqual(config["block_size"], 8)


if __name__ == "__main__":
    unittest.main()
