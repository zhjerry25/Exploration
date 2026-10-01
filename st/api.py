"""Public construction/loading API, independent of CLI and task generators."""
import torch

from .models.baseline import BaselineModel
from .config import ModelConfig, ExecutionConfig
from .models.stack import StackModel
from .runtime import checkpoint


def build_model(config=ModelConfig(), *, execution=ExecutionConfig(), device="cpu", dtype=None):
    config = ModelConfig.from_dict(config)
    if isinstance(execution, dict):
        execution = ExecutionConfig(**execution)
    options = config.to_dict()
    kind = options.pop("model")
    with torch.device(device):
        if kind == "stack":
            model = StackModel(**options, backend=execution.backend,
                               checkpoint_chunks=execution.checkpoint_chunks,
                               encoder_chunk=execution.encoder_chunk,
                               pos_chunk=execution.query_chunk, loss_chunk=execution.loss_chunk)
        else:
            model = BaselineModel(**options, checkpoint_chunks=execution.checkpoint_chunks,
                                  loss_chunk=execution.loss_chunk)
    model.model_config = config
    if dtype is not None:
        model.to(dtype=dtype)
    return model


def load_model(path, *, device="cpu", dtype=None, execution=ExecutionConfig(), topk=None):
    """Load model weights from a trusted new or legacy checkpoint, in eval mode.

    Architecture always comes from the checkpoint. Only inference top-k and
    execution settings may be overridden. Optimizer/RNG restoration belongs
    to the unified training engine, not to this weights-only API.
    """
    state = checkpoint.load(path)
    options = dict(checkpoint.model_config(state))
    if topk is not None:
        if options.get("model", "stack") != "stack":
            raise ValueError("topk is a StackModel inference setting")
        options["topk"] = topk
    # Construct directly on the requested device so a large checkpoint does
    # not transiently hold a second full parameter copy on the host->device
    # transfer path. The checkpoint itself remains mmap-backed CPU state.
    model = build_model(options, execution=execution, device=device)
    model.load_state_dict(state["model"], strict=True)
    if dtype is not None:
        model.to(dtype=dtype)
    return model.eval()
