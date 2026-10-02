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
    # 0 lets StackModel choose a memory-bounded query tile from the model and
    # sequence length.  Baseline does not use this field for its encoder.
    query_chunk: int = 0
    loss_chunk: int = 128
    # Native SDPA mode used by the fair dense baseline.  ``auto`` lets
    # PyTorch select FlashAttention/memory-efficient/math per device.
    flash_attention: str = "auto"
    # Reference attention tile sizes.  Triton kernels choose their own launch
    # tile; these bound the portable torch implementation on long contexts.
    attention_q_chunk: int = 128
    attention_kv_chunk: int = 4096

    def __post_init__(self):
        if self.backend not in ("auto", "torch", "triton"):
            raise ValueError("backend must be auto, torch or triton")
        if self.flash_attention not in ("auto", "flash", "math"):
            raise ValueError("flash_attention must be auto, flash or math")
        values = (self.encoder_chunk, self.loss_chunk,
                  self.attention_q_chunk, self.attention_kv_chunk)
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in values):
            raise ValueError("execution chunk sizes must be positive")
        if isinstance(self.query_chunk, bool) or not isinstance(self.query_chunk, int) or self.query_chunk < 0:
            raise ValueError("query_chunk must be non-negative (0 = auto)")

    def to_dict(self):
        return asdict(self)
