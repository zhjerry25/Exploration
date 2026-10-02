import math
import unittest

import torch

from st import MetricAccumulator, parse_bucket_edges, tail_mask


class MetricTests(unittest.TestCase):
    def test_streaming_metrics_and_buckets(self):
        edges = parse_bucket_edges("0,2,4", 4)
        self.assertEqual(edges, (0, 2, 4))
        metric = MetricAccumulator(edges)
        metric.start_rows(2, track_rows=False)
        logits = torch.tensor([
            [[5., 0., 0.], [0., 5., 0.]],
            [[0., 5., 0.], [0., 0., 5.]],
        ])
        targets = torch.tensor([[0, 1], [1, 2]])
        metric.update(logits, targets, torch.tensor([[0, 1], [2, 3]]))
        result = metric.finalize()
        self.assertEqual(result["evaluated_tokens"], 4)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["ppl"], math.exp(result["loss"]))
        self.assertEqual([b["count"] for b in result["buckets"]], [2, 2])

    def test_tail_mask(self):
        mask = tail_mask((2, 5), 2)
        self.assertEqual(mask.tolist(), [[False, False, False, True, True]] * 2)


if __name__ == "__main__":
    unittest.main()

