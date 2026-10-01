"""Stack experimental framework: stable public Python imports.

Triton is imported lazily only for a CUDA attention call. See docs/API.md.
"""
from .api import build_model, load_model
from .models.baseline import BaselineModel
from .config import ModelConfig, ExecutionConfig
from .runtime.inference import InferenceSession
from .runtime.parallel import ParallelContext
from .models.stack import StackModel
from .data.tokens import TokenDataset

__version__ = "0.2.0"
__all__ = ["StackModel", "BaselineModel", "ModelConfig", "ExecutionConfig",
           "build_model", "load_model", "InferenceSession", "ParallelContext", "TokenDataset"]

# Import aliases preserve previous Python integrations without duplicate files
# or implementations. CLI entry points are exclusively routed by st.cli.
import sys as _sys
from .models import blocks, stack as stack_model, baseline
from .ops import attention, sparse
from .runtime import checkpoint, inference, parallel, memory
from .data import tokens as token_data
for _name in ("blocks", "stack_model", "baseline", "attention", "sparse",
              "checkpoint", "inference", "parallel", "memory", "token_data"):
    _sys.modules[f"{__name__}.{_name}"] = globals()[_name]
del _sys, _name
