"""Unified driver/API regressions, including legacy checkpoint migration."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from st import ModelConfig, ExecutionConfig, StackModel, build_model, load_model
from st import checkpoint
from st.cli import main, parse_args, entrypoint


class UnifiedDriverTests(unittest.TestCase):
    def test_reorganized_module_aliases(self):
        import st.blocks
        import st.stack_model
        import st.attention
        from st.models import blocks, stack
        from st.ops import attention
        from st import data
        self.assertIs(st.blocks, blocks)
        self.assertIs(st.stack_model.StackModel, stack.StackModel)
        self.assertIs(st.attention.dense_attention, attention.dense_attention)
        self.assertEqual(data.Q, 31)
        self.assertEqual(data.KEY, 5)

    def test_command_help_is_scoped_and_no_args_shows_help(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            main([])
        self.assertIn("benchmark", out.getvalue())
        for command, present, absent in (("eval", "--cache", "--grad-accum"),
                                         ("plan", "--dim", "--eval-every"),
                                         ("benchmark", "--compare", "--resume")):
            with self.subTest(command=command), contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as caught:
                    main([command, "--help"])
                self.assertEqual(caught.exception.code, 0)
            self.assertIn(present, out.getvalue())
            self.assertNotIn(absent, out.getvalue())

    def test_console_entrypoint_discards_result(self):
        with patch("st.cli.main", return_value={"loss": 1.0}) as driver:
            self.assertIsNone(entrypoint())
        driver.assert_called_once_with()

    def test_json_defaults_are_validated(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"config.json"
            for runtime in ({"precision": "fp64"}, {"batch_size": 1.5},
                            {"checkpoint_chunks": "false"}, {"command": "eval"}):
                path.write_text(json.dumps({"runtime": runtime}))
                with self.subTest(runtime=runtime), contextlib.redirect_stderr(io.StringIO()), self.assertRaises((SystemExit, ValueError)):
                    parse_args(["plan", "--config", str(path)])

    def cpu(self):
        return ["--device", "cpu", "--backend", "torch", "--precision", "fp32"]

    def test_public_build_load_and_legacy_checkpoint(self):
        config = ModelConfig(vocab_size=128, dim=16, heads=2, block_size=4, arch="L,G")
        model = build_model(config, execution=ExecutionConfig(backend="torch", encoder_chunk=8))
        self.assertIsInstance(model, StackModel)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"legacy.pt"
            torch.save({"model": model.state_dict(), "args": {"model": "stack", "d": 16,
                        "heads": 2, "b": 4, "arch": "L,G", "task": "mqar", "read_m": 64}}, path)
            other = load_model(path, topk=2, execution=ExecutionConfig(backend="torch", encoder_chunk=8))
            self.assertEqual(other.topk, 2)
            self.assertEqual(other.dim, 16)
            self.assertFalse(other.training)
            for p, q in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_train_eval_baseline_and_stack_share_one_driver(self):
        for kind in ("stack", "baseline"):
            with self.subTest(model=kind), tempfile.TemporaryDirectory() as td:
                ck = str(Path(td)/"model.pt")
                common = ["--model", kind, "--task", "random", "--vocab-size", "31", "--dim", "16",
                          "--heads", "2", "--block-size", "4", "--arch", "L,G", "--layers", "1",
                          "--length", "17", "--batch-size", "2", "--encoder-chunk", "8", "--query-chunk", "3"]
                with contextlib.redirect_stdout(io.StringIO()):
                    main(["train", *common, *self.cpu(), "--steps", "2", "--grad-accum", "2",
                          "--eval-every", "1", "--save", ck])
                    result = main(["eval", "--resume", ck, "--task", "random", "--length", "23",
                                   "--eval-positions", "3", "--cache", "cpu", *self.cpu()])
                saved = checkpoint.load(ck)
                self.assertEqual(saved["config"]["model"]["model"], kind)
                self.assertEqual(saved["step"], 1)
                self.assertEqual(result["evaluated_tokens"], 3)
                self.assertTrue(torch.isfinite(torch.tensor(result["loss"])))
                self.assertNotIn("ema", saved)

    def test_config_round_trip_and_checkpoint_authority(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/"config.json"
            path.write_text(json.dumps({"model": {"dim": 32, "heads": 4},
                                        "runtime": {"batch_size": 2}}))
            args = parse_args(["plan", "--config", str(path), "--dim", "16"])
            self.assertEqual(args.dim, 16)
            self.assertEqual(args.batch_size, 2)
            with contextlib.redirect_stdout(io.StringIO()) as output:
                main(["plan", "--config", str(path), "--dim", "16"])
            self.assertEqual(json.loads(output.getvalue())["model"]["dim"], 16)

    def test_dense_loss_does_not_depend_on_inference_topk(self):
        torch.manual_seed(14)
        small = build_model(ModelConfig(vocab_size=31, dim=16, heads=2, block_size=4, topk=1),
                            execution=ExecutionConfig(backend="torch", encoder_chunk=8))
        large = build_model(ModelConfig(vocab_size=31, dim=16, heads=2, block_size=4, topk=64),
                            execution=ExecutionConfig(backend="torch", encoder_chunk=8))
        large.load_state_dict(small.state_dict())
        ids = torch.randint(31, (1, 21))
        targets = torch.randint(31, ids.shape)
        torch.testing.assert_close(small(ids, targets=targets)["loss_sum"],
                                   large(ids, targets=targets)["loss_sum"], atol=0, rtol=0)

    def test_invalid_cli_and_old_ema_flag_are_rejected(self):
        for argv in (["train", "--length", "65537"], ["train", "--topk", "0"],
                     ["train", "--stop-exact", ".99"], ["train", "--ema", ".9"],
                     ["train", "--task", "tokens", "--eval-every", "1"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)


if __name__ == "__main__":
    unittest.main()
