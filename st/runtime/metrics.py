"""Small, streaming metric accumulators shared by train and eval.

The evaluator never needs to concatenate long-context logits.  It updates
token totals and optional absolute-position buckets as each query chunk is
produced, then returns the common LM (loss, bpc, perplexity, accuracy) and
row-level exact-match metrics.
"""
from dataclasses import dataclass, field
import math

import torch
from torch.nn import functional as F


def tail_mask(shape_or_targets, tail_tokens):
    """Return a bool mask supervising only the final ``tail_tokens`` columns."""
    shape = shape_or_targets.shape if hasattr(shape_or_targets, "shape") else tuple(shape_or_targets)
    if len(shape) != 2 or tail_tokens < 1:
        raise ValueError("tail supervision requires [batch, length] and a positive tail_tokens")
    mask = torch.zeros(shape, dtype=torch.bool,
                       device=shape_or_targets.device if hasattr(shape_or_targets, "device") else None)
    mask[:, max(0, shape[1] - int(tail_tokens)):] = True
    return mask


def default_bucket_edges(length, count=4):
    """Evenly spaced absolute-position edges, including 0 and ``length``."""
    if length < 1 or count < 1:
        raise ValueError("length and bucket count must be positive")
    edges = sorted(set([0, length] + [round(length*i/count) for i in range(1, count)]))
    return tuple(int(x) for x in edges)


def parse_bucket_edges(spec, length):
    """Parse ``'0,1024,4096'``; an empty spec selects four even buckets."""
    if spec is None or spec == "" or spec == "auto":
        return default_bucket_edges(length)
    try:
        edges = tuple(int(item.strip()) for item in str(spec).split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("metric buckets must be comma-separated integers") from exc
    if len(edges) < 2 or edges[0] != 0 or edges[-1] != length or any(a >= b for a, b in zip(edges, edges[1:])):
        raise ValueError("metric buckets must be increasing and start at 0/end at length")
    return edges


@dataclass
class MetricAccumulator:
    """Streaming token and row metrics for a fixed evaluation batch."""

    bucket_edges: tuple | None = None
    loss_sum: float = 0.0
    count: int = 0
    correct: int = 0
    row_errors: torch.Tensor | None = None
    _bucket_loss: list = field(default_factory=list)
    _bucket_count: list = field(default_factory=list)
    _bucket_correct: list = field(default_factory=list)

    def start_rows(self, batch, device="cpu", track_rows=True):
        self.row_errors = (torch.zeros(batch, dtype=torch.long, device=device)
                           if track_rows else None)
        if self.bucket_edges is not None:
            n = len(self.bucket_edges) - 1
            self._bucket_loss = [0.0] * n
            self._bucket_count = [0] * n
            self._bucket_correct = [0] * n

    def update(self, logits, targets, positions=None):
        if logits.ndim != 3 or targets.shape != logits.shape[:2]:
            raise ValueError("logits must be [batch, queries, vocab] and targets [batch, queries]")
        per_token = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
        hits = logits.argmax(-1).eq(targets)
        self.loss_sum += float(per_token.detach().sum())
        self.count += int(targets.numel())
        self.correct += int(hits.sum())
        if self.row_errors is not None:
            self.row_errors += (~hits).sum(1).to(self.row_errors.device)
        if self.bucket_edges is not None and positions is not None:
            if positions.shape != targets.shape:
                raise ValueError("positions must match targets for bucket metrics")
            edges = torch.tensor(self.bucket_edges, device=positions.device)
            # ``right=True`` makes boundaries left-inclusive: [lo, hi).
            bucket = torch.bucketize(positions.contiguous(), edges[1:-1], right=True)
            for index in range(len(self._bucket_count)):
                selected = bucket == index
                self._bucket_loss[index] += float(per_token[selected].detach().sum())
                self._bucket_count[index] += int(selected.sum())
                self._bucket_correct[index] += int(hits[selected].sum())

    def finalize(self):
        if self.count < 1:
            raise ValueError("cannot finalize metrics with zero tokens")
        loss = self.loss_sum / self.count
        result = {
            "loss": loss,
            "bpc": loss / math.log(2.0),
            "ppl": math.exp(min(loss, 80.0)),
            "accuracy": self.correct / self.count,
            "evaluated_tokens": self.count,
        }
        if self.row_errors is not None:
            result["exact"] = float((self.row_errors == 0).sum()) / max(1, self.row_errors.numel())
        if self.bucket_edges is not None:
            buckets = []
            for lo, hi, loss_sum, count, correct in zip(
                    self.bucket_edges, self.bucket_edges[1:], self._bucket_loss,
                    self._bucket_count, self._bucket_correct):
                buckets.append({
                    "start": lo, "end": hi, "count": count,
                    "loss": loss_sum / count if count else None,
                    "bpc": loss_sum / count / math.log(2.0) if count else None,
                    "ppl": math.exp(min(loss_sum / count, 80.0)) if count else None,
                    "accuracy": correct / count if count else None,
                })
            result["buckets"] = buckets
        return result
