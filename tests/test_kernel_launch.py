"""Launch-policy regression without importing Triton or executing a GPU."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


class LaunchPolicyTests(unittest.TestCase):
    def policy(self):
        # Execute only the pure Python launch policy; CUDA calls are replaced
        # with a fake JIT launch to verify retry/exception contracts.
        path = Path(__file__).resolve().parents[1]/"st/ops/kernels/launch.py"
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        class OutOfResources(Exception):
            pass
        namespace = {"torch": SimpleNamespace(device=lambda value: SimpleNamespace(index=0), float32="fp32"),
                     "OutOfResources": OutOfResources}
        exec(compile(tree, str(path), "exec"), namespace)
        return namespace

    def test_fp32_starts_with_one_stage_and_retries_only_resources(self):
        policy = self.policy()
        calls = []
        class Kernel:
            __name__ = "fake_forward"
            def __getitem__(self, grid):
                def call(*args, **meta):
                    calls.append(meta)
                    if len(calls) == 1:
                        raise policy["OutOfResources"]("shared memory")
                    return SimpleNamespace(metadata=SimpleNamespace(shared=32000))
                return call
        meta = dict(SQ=5, SK=193, H=2, D=96, DM=128, B=64, M=16, N=64)
        result = policy["launch"](Kernel(), lambda c: (1,), (), meta, "cuda:0", "fp32")
        self.assertEqual(calls[0]["num_stages"], 1)
        self.assertEqual(result["num_warps"], 8)
        self.assertEqual(policy["launch_report"]()[0]["shared_memory_bytes"], 32000)
        count = len(calls)
        policy["launch"](Kernel(), lambda c: (1,), (), meta, "cuda:0", "fp32")
        self.assertEqual(len(calls), count+1)

    def test_unrelated_errors_are_not_swallowed(self):
        policy = self.policy()
        class Kernel:
            __name__ = "bad_kernel"
            def __getitem__(self, grid):
                def call(*args, **meta):
                    raise ValueError("compile failure")
                return call
        with self.assertRaisesRegex(ValueError, "compile failure"):
            policy["launch"](Kernel(), lambda c: (1,), (), {"N": 64, "B": 16}, "cuda:0", "fp32")

    def test_exhausted_matrix_configuration_is_cached(self):
        policy = self.policy()
        calls = []
        class Kernel:
            __name__ = "large_matrix"
            def __getitem__(self, grid):
                def call(*args, **meta):
                    calls.append(meta)
                    raise policy["OutOfResources"]("139264 > 101376")
                return call
        meta = dict(N=128, B=128, D=128)
        with self.assertRaises(policy["ResourceExhausted"]):
            policy["launch"](Kernel(), lambda c: (1,), (), meta, "cuda:0", "fp32")
        count = len(calls)
        with self.assertRaises(policy["ResourceExhausted"]):
            policy["launch"](Kernel(), lambda c: (1,), (), meta, "cuda:0", "fp32")
        self.assertEqual(len(calls), count)
        self.assertTrue(all(c["N"] >= c["B"] for c in calls))

    def test_streamed_tile_can_be_smaller_than_model_block(self):
        policy = self.policy()
        calls = []
        class Kernel:
            __name__ = "streamed"
            def __getitem__(self, grid):
                def call(*args, **meta):
                    calls.append(meta)
                    if meta["T"] > 16:
                        raise policy["OutOfResources"]("test tile budget")
                    return SimpleNamespace(metadata=SimpleNamespace(shared=2048))
                return call
        result = policy["launch_streamed"](Kernel(), lambda c: (1,), (), {"B": 128, "D": 256}, "cuda:0", "fp32")
        self.assertEqual(result["T"], 16)
        self.assertEqual(policy["launch_report"]()[0]["path"], "streamed_reduction")
