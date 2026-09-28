"""Driver-level regression: a checkpoint must be authoritative for model
structure. Eval/resume with CLI structure flags that MISMATCH the trained
configuration has to succeed (this bug killed the E2 extrapolation sweep)."""
import argparse
import os
import tempfile
import unittest

import torch

from st import train as driver


def cli_args(**over):
    args = argparse.Namespace(
        model="stack", task="mqar", n=64, b=4, d=32, heads=2, arch=None,
        layers=3, ffn_ratio=4.0, bs=4, steps=2, lr=1e-3, eval_every=1,
        ema=0.0, bf16=False, save="", ckpt_every=0, resume="",
        resume_weights_only="", seed=0, read_m=None, nqueries=2, npairs=2,
        npairs_density=0.0, nkeytoks=1, stop_exact=None, tag=None,
        eval_only=False, prof=0, device="cpu")
    for k, v in over.items():
        setattr(args, k, v)
    return args


class TrainDriverTests(unittest.TestCase):
    def test_resume_inherits_checkpoint_structure(self):
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            # train 2 steps with a NON-default arch (L,G) and block size 4
            train = cli_args(arch="L,G", save="ck.pt", tag="t")
            driver.train(train, torch.device("cpu"))
            # eval with CLI defaults (arch Lx2,G, b=16): must still load
            ev = cli_args(arch=None, b=16, n=128, steps=1, eval_only=True,
                          resume="ck.pt", tag="e")
            driver.train(ev, torch.device("cpu"))  # no exception = structure inherited
            # read_m override remains an inference-time knob
            self.assertEqual(ev.read_m, 64)
            # and --read_m explicitly set wins over the checkpoint's
            ev2 = cli_args(eval_only=True, resume="ck.pt", read_m=2, tag="e2")
            driver.train(ev2, torch.device("cpu"))
            self.assertEqual(ev2.read_m, 2)

    def test_doctrine_gate_rejects_sparse_training(self):
        with tempfile.TemporaryDirectory() as td:
            os.chdir(td)
            with self.assertRaises(SystemExit):
                driver.train(cli_args(n=256, b=4, read_m=8), torch.device("cpu"))


if __name__ == '__main__':
    unittest.main()
