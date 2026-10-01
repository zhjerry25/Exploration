"""Stable, serializable public model and execution configuration."""
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class ModelConfig:
    model: str = "stack"
    vocab_size: int = 128
    dim: int = 256
    heads: int = 4
    block_size: int = 16
    topk: int = 64
    arch: str = "Lx2,G"
    ffn_ratio: float = 4.
    layers: int = 3

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        unknown = set(value)-{f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown model configuration keys: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        result = asdict(self)
        if self.model == "stack":
            result.pop("layers")
        elif self.model == "baseline":
            for key in ("arch", "block_size", "topk"):
                result.pop(key)
        else:
            raise ValueError("model must be stack or baseline")
        return result


@dataclass(frozen=True)
class ExecutionConfig:
    backend: str = "auto"
    checkpoint_chunks: bool = True
    encoder_chunk: int = 1024
    query_chunk: int = 128
    loss_chunk: int = 128

    def __post_init__(self):
        if self.backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be auto, torch or triton")
        if min(self.encoder_chunk, self.query_chunk, self.loss_chunk) < 1:
            raise ValueError("execution chunk sizes must be positive")

    def to_dict(self):
        return asdict(self)
