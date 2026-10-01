"""Streaming prefill and paged, optionally multi-GPU sparse inference.

Keep input IDs on CPU. Only a page of hidden states and raw KV is computed
at a time. Encoded query states are kept ONLY for requested positions.
GPU/host/disk cache tiers share the exact same attention semantics.
"""
import os
from pathlib import Path
import tempfile
import shutil

import torch
import torch.distributed as dist

from .memory import plan_inference
from .sparse import TensorPages, sparse_attention


class KVCache(TensorPages):
    def __init__(self, batch, length, heads, dim, dtype, device, plan, offset=0,
                 cache_dir=None):
        self.directory = None
        shape = (batch, length, heads, dim)
        if plan.cache == "disk":
            self.directory = tempfile.TemporaryDirectory(prefix="stack-kv-", dir=cache_dir)
            tensors = []
            count = batch*length*heads*dim
            for name in ("keys", "values"):
                path = Path(self.directory.name)/f"{name}.bin"
                # Allocate real disk blocks where supported. Fail here, not
                # via SIGBUS half-way through a multi-hour prefill.
                with open(path, "xb") as file:
                    size = count*torch.empty((), dtype=dtype).element_size()
                    if hasattr(os, "posix_fallocate"):
                        os.posix_fallocate(file.fileno(), 0, size)
                    else:
                        file.truncate(size)
                tensors.append(torch.from_file(str(path), shared=True, size=count, dtype=dtype).reshape(shape))
            k, v = tensors
        else:
            target = device if plan.cache == "cuda" else "cpu"
            k, v = (torch.empty(shape, dtype=dtype, device=target) for _ in range(2))
        super().__init__(k, v, plan.page_tokens, offset)

    def write(self, start, k, v):
        lo = start-self.offset
        if lo < 0 or lo+k.shape[1] > self.length:
            raise ValueError("KV write outside owned shard")
        self.k[:, lo:lo+k.shape[1]].copy_(k)
        self.v[:, lo:lo+v.shape[1]].copy_(v)

    def close(self):
        self.k, self.v = None, None
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class InferenceSession:
    """A write-once raw-KV cache and bounded query states for one prompt.

    ``prefill(ids_cpu, positions_cpu)`` followed by ``iter_logits()``.
    All ranks in group use identical ids and positions. Use a context
    manager to release caches deterministically, including mmap files.
    This is prompt evaluation; it does not pretend to be an append/decode
    API. Creating a new prompt requires a new session.
    """
    def __init__(self, model, *, group=None, cache="auto", cache_dir=None,
                 memory_fraction=.8, page_tokens=65536, encoder_chunk=4096,
                 query_chunk=16, workspace_mb=256, max_query_states_mb=256):
        if model.training:
            raise ValueError("call model.eval() before creating an inference session")
        self.model, self.group = model, group
        self.device, self.dtype = next(model.parameters()).device, next(model.parameters()).dtype
        self.world = dist.get_world_size(group) if group is not None else 1
        self.rank = dist.get_rank(group) if group is not None else 0
        if not 1 <= self.world <= 8:
            raise ValueError("inference supports 1..8 GPUs")
        self.options = dict(cache=cache, cache_dir=cache_dir, memory_fraction=memory_fraction,
                            page_tokens=page_tokens, encoder_chunk=encoder_chunk,
                            query_chunk=query_chunk, workspace_mb=workspace_mb)
        self.max_query_bytes = max_query_states_mb*(1 << 20)
        self.cache = None
        self.query_directory = None
        self.ready = False

    @torch.no_grad()
    def prefill(self, input_ids, positions):
        if self.cache is not None or self.ready:
            raise RuntimeError("use a new session for each prompt")
        if input_ids.ndim != 2 or min(input_ids.shape) < 1 or input_ids.device.type != "cpu":
            raise ValueError("input_ids must be nonempty CPU [B,N]; pages are transferred automatically")
        batch, n = input_ids.shape
        positions = positions.to("cpu", dtype=torch.long)
        if positions.ndim != 2 or positions.shape[0] != batch or positions.shape[1] < 1:
            raise ValueError("positions must be nonempty [B,Q]")
        if (positions < 0).any() or (positions >= n).any():
            raise ValueError("query positions must lie within the prompt")
        query_bytes = positions.numel()*self.model.dim*torch.empty((), dtype=self.dtype).element_size()
        query_disk = query_bytes > self.max_query_bytes
        if query_disk and not self.options["cache_dir"]:
            raise MemoryError("query states exceed --max-query-states-mb; provide --cache-dir for paged query states or evaluate fewer positions")
        b = self.model.block_size
        shard = ((n+self.world*b-1)//(self.world*b))*b
        start, end = self.rank*shard, min((self.rank+1)*shard, n)
        if start >= end:
            raise ValueError("each inference rank must own at least one token block")
        self.plan = plan_inference(self.model, batch, end-start, **self.options)
        self.cache = KVCache(batch, end-start, self.model.heads, self.model.hd,
                             self.dtype, self.device, self.plan, start, self.options["cache_dir"])
        self.positions = positions
        # Unsorted and duplicate requests retain their original order. The
        # sorted metadata makes prefill lookup O(log Q + page_queries), not
        # an O(Q) scan for every encoded page.
        sorted_pos, sorted_index = positions.sort(1)
        # Recompute the exact finite receptive-field warmup independently on
        # each rank. At L depth, only L preceding blocks can affect x[start].
        warm = max(0, start-len(self.model.local)*b)
        states = [None]*len(self.model.local)
        try:
            if query_disk:
                directory = self.options["cache_dir"]
                local_processes = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
                if query_bytes*local_processes+(1 << 30) > shutil.disk_usage(directory).free:
                    raise MemoryError("insufficient disk space for query-state cache")
                self.query_directory = tempfile.TemporaryDirectory(prefix="stack-queries-", dir=directory)
                path = Path(self.query_directory.name)/"queries.bin"
                with open(path, "xb") as file:
                    if hasattr(os, "posix_fallocate"):
                        os.posix_fallocate(file.fileno(), 0, query_bytes)
                    else:
                        file.truncate(query_bytes)
                self.query_states = torch.from_file(str(path), shared=True, size=positions.numel()*self.model.dim,
                                                     dtype=self.dtype).reshape(batch, positions.shape[1], self.model.dim)
                # Newly allocated files read as zero; unowned query rows
                # must remain zero before cross-rank reduction.
            else:
                self.query_states = torch.zeros(batch, positions.shape[1], self.model.dim, dtype=self.dtype, device="cpu")
            for lo in range(warm, end, self.plan.encoder_chunk):
                hi = min(end, lo+self.plan.encoder_chunk)
                x = self.model.embedding(input_ids[:, lo:hi].to(self.device))
                for i, layer in enumerate(self.model.local):
                    prev = states[i]
                    states[i] = x[:, -b:].clone()
                    x = layer(x, previous=prev, start=lo)
                k, v = self.model.raw_kv(x, None)
                keep = max(start, lo)
                if hi > keep:
                    self.cache.write(keep, k[:, keep-lo:], v[:, keep-lo:])
                    for row in range(batch):
                        left, right = torch.searchsorted(sorted_pos[row], torch.tensor([keep, hi])).tolist()
                        if right > left:
                            cols = sorted_index[row, left:right]
                            gathered = x[row, (positions[row, cols]-lo).to(self.device)]
                            self.query_states[row, cols] = gathered.cpu()
                del x, k, v
            self.ready = True
        except BaseException:
            self.close()
            raise
        return self

    @torch.no_grad()
    def iter_logits(self, return_selection=False):
        if not self.ready:
            raise RuntimeError("prefill must complete before inference")
        m = self.model
        for lo in range(0, self.positions.shape[1], self.plan.query_chunk):
            hi = min(lo+self.plan.query_chunk, self.positions.shape[1])
            z = self.query_states[:, lo:hi].to(self.device)
            if self.world > 1:
                dist.all_reduce(z, group=self.group)
            pos = self.positions[:, lo:hi].to(self.device)
            selection = None
            for rd in m.reads:
                q = rd.wq(rd.q_norm(z)).reshape(*z.shape[:2], m.heads, m.hd)
                result = sparse_attention(q, self.cache, pos, m.block_size, m.topk,
                    backend=m.backend, group=self.group, score_page_tokens=self.plan.score_tokens,
                    return_selection=return_selection)
                if return_selection:
                    ctx, selection = result
                else:
                    ctx = result
                z = z + rd.read_out(ctx.reshape(*z.shape[:2], m.dim))
                z = z + rd.ffn(rd.ffn_norm(z))
            for j in range(0, hi-lo, self.plan.head_chunk):
                stop = min(j+self.plan.head_chunk, hi-lo)
                logits = m.lm_head(m.final_norm(z[:, j:stop]))
                # Consumer decides whether to save logits, metrics, or only
                # argmax. No implicit dense [B,N,V] allocation.
                yield slice(lo+j, lo+stop), logits, selection

    def close(self):
        if self.cache is not None:
            self.cache.close()
            self.cache = None
        self.query_states = None
        if self.query_directory is not None:
            self.query_directory.cleanup()
            self.query_directory = None
        self.ready = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
