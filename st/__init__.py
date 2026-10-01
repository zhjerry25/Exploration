"""Stack experimental framework: stable public Python imports.

Triton is imported lazily only for a CUDA attention call. See docs/API.md.
"""
from .api import build_model, load_model
from .baseline import BaselineModel
from .config import ModelConfig, ExecutionConfig
from .inference import InferenceSession
from .parallel import ParallelContext
from .stack_model import StackModel
from .token_data import TokenDataset

__version__ = "0.2.0"
__all__ = ["StackModel", "BaselineModel", "ModelConfig", "ExecutionConfig",
           "build_model", "load_model", "InferenceSession", "ParallelContext", "TokenDataset"]
