"""Portable FlashAttention dispatch through PyTorch SDPA.

The baseline deliberately uses this module instead of hand-written score
matrices.  ``scaled_dot_product_attention`` selects the best native kernel
(FlashAttention, memory-efficient, or math) for the current device and dtype.
The explicit ``flash`` mode is useful for validation; the default ``auto``
mode keeps the public framework compatible with CPU, older CUDA, and MPS.
"""
from contextlib import nullcontext

import torch
from torch.nn import functional as F


def _sdpa_context(mode):
    if mode == "auto":
        return nullcontext()
    if mode not in ("flash", "math"):
        raise ValueError("attention mode must be auto, flash or math")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # PyTorch 2.4 and older
        if mode == "flash":
            return torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False,
                enable_mem_efficient=True)
        return torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True,
            enable_mem_efficient=False)
    backend = SDPBackend.FLASH_ATTENTION if mode == "flash" else SDPBackend.MATH
    return sdpa_kernel(backend)


def flash_attention(q, k, v, *, is_causal=False, attn_mask=None,
                    dropout_p=0.0, mode="auto", enable_gqa=False):
    """Run attention with the native fused SDPA implementation.

    ``q/k/v`` use the usual ``[batch, heads, sequence, head_dim]`` layout.
    No score matrix is materialized by the FlashAttention and memory-efficient
    kernels.  ``mode='auto'`` is the recommended setting for heterogeneous
    deployments; ``mode='flash'`` fails loudly when a device cannot satisfy
    the fused-kernel constraints.
    """
    if dropout_p < 0:
        raise ValueError("dropout_p must be non-negative")
    with _sdpa_context(mode):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, enable_gqa=enable_gqa)


def backend_info(device=None, dtype=None):
    """Return a serializable description of the native SDPA capabilities."""
    device = torch.device(device or "cpu")
    info = {"device": str(device), "dtype": str(dtype) if dtype is not None else None,
            "implementation": "torch-scaled-dot-product-attention"}
    if device.type == "cuda":
        info.update({
            "flash_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
        })
        try:
            info["flash_available"] = bool(torch.backends.cuda.is_flash_attention_available())
        except AttributeError:
            info["flash_available"] = None
    else:
        info.update({"flash_enabled": False, "mem_efficient_enabled": False,
                     "math_enabled": True, "flash_available": False})
    return info

