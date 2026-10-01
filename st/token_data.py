"""Memory-mapped token corpus, no hidden downloads or full-corpus int64 copy."""
from pathlib import Path
import torch


class TokenDataset:
    def __init__(self, path, dtype="uint16", start=0, stop=None):
        types = {"uint8": torch.uint8, "uint16": torch.uint16, "int32": torch.int32, "int64": torch.int64}
        if dtype not in types:
            raise ValueError("token dtype must be uint8/uint16/int32/int64")
        self.path = Path(path)
        dt = types[dtype]
        width = torch.empty((), dtype=dt).element_size()
        size = self.path.stat().st_size
        if size % width:
            raise ValueError("token file byte count is not divisible by token dtype size")
        total = size//width
        stop = total if stop is None else stop
        if not 0 <= start < stop <= total:
            raise ValueError("invalid corpus token interval")
        self.tokens = torch.from_file(str(self.path), shared=False, size=total, dtype=dt)[start:stop]

    def batch(self, batch_size, length, generator):
        if len(self.tokens) < length+1:
            raise ValueError(f"corpus has {len(self.tokens)} tokens but context requires {length+1}")
        offsets = torch.randint(len(self.tokens)-length, (batch_size,), generator=generator)
        # Slice uint16 before conversion: some torch backends have limited
        # uint16 advanced-indexing support.
        seq = torch.stack([self.tokens[int(i):int(i)+length+1].long() for i in offsets])
        return seq[:, :-1], seq[:, 1:], torch.ones(batch_size, length, dtype=torch.bool), None
