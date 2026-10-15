"""Explicit CUDA candidate discovery without early numerical imports."""

from . import execution


def native_cuda(device):
    """Require the requested native CUDA device, with no CPU fallback."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    try:
        from cgpy import cuda
    except ImportError as exc:
        raise RuntimeError(
            "CUDA requested but the native CUDA extension is unavailable"
        ) from exc
    if not cuda.enabled() or not cuda.is_available(device):
        raise RuntimeError(
            f"CUDA requested but native CUDA device {device} is unavailable"
        )
    return cuda


def torch_cuda(device):
    """Require CUDA PyTorch and a valid logical device ordinal."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA requested but PyTorch is unavailable"
        ) from exc
    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA requested but PyTorch CUDA device {device} is unavailable"
        )
    return torch


def preflight_cuda(names, device):
    """Fail before collecting or writing timings if any requested GPU is absent."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    if not names or any(
        name not in execution.CUDA_IMPLEMENTATIONS for name in names
    ):
        raise ValueError("CUDA comparison requires explicit CUDA candidates")
    if any(name in ("cuda-native", "cuda-cublas") for name in names):
        native_cuda(device)
    if "torch-cuda-eager" in names:
        torch_cuda(device)
