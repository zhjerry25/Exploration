"""1..8 GPU data/context parallelism (Ulysses sequence/head all-to-all).

Encoder activations and queries are sequence-sharded. Global attention owns
the full sequence of only H/CP heads. Each tensor has O(B N D / CP) storage.
One-block differentiable halos preserve each local layer's exact receptive
field; shared/cycled layers exchange a fresh halo on EVERY application.
"""
from dataclasses import dataclass
import os

import torch
import torch.distributed as dist
from torch.autograd import Function


def _exchange(x, group, inverse):
    size = dist.get_world_size(group)
    batch, length, heads, dim = x.shape
    if inverse:
        send = x.reshape(batch, size, length // size, heads, dim).permute(1, 0, 2, 3, 4).contiguous()
    else:
        send = x.reshape(batch, length, size, heads // size, dim).permute(2, 0, 1, 3, 4).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    if inverse:
        return recv.permute(1, 2, 0, 3, 4).reshape(batch, length // size, heads * size, dim).contiguous()
    return recv.permute(1, 0, 2, 3, 4).reshape(batch, length * size, heads // size, dim).contiguous()


class _AllToAll(Function):
    @staticmethod
    def forward(ctx, x, group, inverse):
        ctx.group, ctx.inverse = group, inverse
        return _exchange(x, group, inverse)

    @staticmethod
    def backward(ctx, grad):
        return _exchange(grad, ctx.group, not ctx.inverse), None, None


class _Halo(Function):
    @staticmethod
    def forward(ctx, tail, group):
        rank, size = dist.get_rank(group), dist.get_world_size(group)
        ctx.group, ctx.rank, ctx.size = group, rank, size
        buffers = [torch.empty_like(tail) for _ in range(size)]
        dist.all_gather(buffers, tail.contiguous(), group=group)
        return buffers[rank-1] if rank else torch.zeros_like(tail)

    @staticmethod
    def backward(ctx, grad):
        # Small halo only; avoids backend-specific reduce_scatter support.
        buf = grad.new_zeros(ctx.size, *grad.shape)
        if ctx.rank:
            buf[ctx.rank-1] = grad
        dist.all_reduce(buf, group=ctx.group)
        return buf[ctx.rank], None


@dataclass
class ParallelContext:
    world_size: int = 1
    rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    cp_group: object = None

    @classmethod
    def initialize(cls, cp_size=1, device="cuda"):
        world = int(os.environ.get("WORLD_SIZE", "1"))
        if not 1 <= world <= 8 or cp_size < 1 or world % cp_size:
            raise ValueError("require 1..8 processes and context_parallel dividing WORLD_SIZE")
        if device == "cuda":
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if world == 1:
            return cls()
        if not dist.is_initialized():
            dist.init_process_group("nccl" if device == "cuda" else "gloo")
        if dist.get_world_size() != world:
            raise ValueError("initialized process group differs from WORLD_SIZE")
        rank = dist.get_rank()
        group = None
        for dp in range(world // cp_size):
            ranks = list(range(dp * cp_size, (dp + 1) * cp_size))
            g = dist.new_group(ranks)
            if rank in ranks:
                group = g
        return cls(world, rank, cp_size, rank % cp_size,
                   world // cp_size, rank // cp_size, group)

    def shard(self, tensor, block_size, value=0):
        """CPU or GPU [B,N,...] -> equal, block-aligned local sequence."""
        n = tensor.shape[1]
        length = ((n + block_size * self.cp_size - 1) // (block_size * self.cp_size)) * block_size
        start = self.cp_rank * length
        result = tensor[:, start:min(start + length, n)]
        if result.shape[1] < length:
            shape = (tensor.shape[0], length - result.shape[1], *tensor.shape[2:])
            result = torch.cat((result, tensor.new_full(shape, value)), 1)
        return result.contiguous(), start

    def halo(self, x, block_size):
        if self.cp_size == 1:
            return None
        # Rank zero must ALSO participate in backward collectives. Its
        # returned halo is masked by the local attention's global boundary.
        return _Halo.apply(x[:, -block_size:], self.cp_group)

    def to_heads(self, x):
        if self.cp_size == 1:
            return x
        if x.shape[2] % self.cp_size:
            raise ValueError("heads must be divisible by context_parallel")
        return _AllToAll.apply(x, self.cp_group, False)

    def to_sequence(self, x):
        return x if self.cp_size == 1 else _AllToAll.apply(x, self.cp_group, True)

    def gather_positions(self, positions):
        if self.cp_size == 1:
            return positions
        out = [torch.empty_like(positions) for _ in range(self.cp_size)]
        dist.all_gather(out, positions.contiguous(), group=self.cp_group)
        return torch.cat(out, 1)

    def max_query_count(self, count, device):
        if self.cp_size == 1:
            return max(1, count)
        c = torch.tensor(count, device=device, dtype=torch.long)
        dist.all_reduce(c, op=dist.ReduceOp.MAX, group=self.cp_group)
        return max(1, int(c))


def sum_all(value):
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
    return result
