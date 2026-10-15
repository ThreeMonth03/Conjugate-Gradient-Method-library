"""Benchmark result contracts and a single explicit candidate factory."""

import dataclasses
from typing import Protocol

from . import validation

CUDA_IMPLEMENTATIONS = ("cuda-native", "cuda-cublas", "torch-cuda-eager")


@dataclasses.dataclass
class Outcome:
    x: object
    status: str
    iterations: int
    matvecs: int
    residual_norm: float

    @property
    def converged(self):
        return self.status == "converged"


class Executor(Protocol):
    def __call__(self) -> Outcome: ...
    def benchmark_metadata(self) -> dict[str, object]: ...


def make_executor(
    name, arrays, options, scope, threads, *, device="cpu", cuda_device=0
) -> Executor:
    validation.validate_scope(scope, device)
    validation.validate_execution_mode(options, device)
    if device == "cpu" and "execution_mode" in options:
        options = {
            key: value
            for key, value in options.items()
            if key != "execution_mode"
        }
    from .adapters import native
    from .reference import engine, numpy, torch
    from .reference import scipy as scipy_ops

    if name in CUDA_IMPLEMENTATIONS:
        if device != "cuda":
            raise ValueError("CUDA candidates require --device cuda")
        from .adapters import cuda

        if name == "torch-cuda-eager":
            return cuda.TorchCudaExecutor(
                arrays, options, scope, threads, cuda_device
            )
        return cuda.CudaNativeExecutor(
            arrays,
            options,
            scope,
            cuda_device,
            "cublas" if name == "cuda-cublas" else "native",
        )
    if device != "cpu":
        raise ValueError(
            "CPU candidates require --device cpu; CUDA never falls back to CPU"
        )
    if name == "native-blas":
        if hasattr(arrays[0], "indptr"):
            raise ValueError("native-blas requires dense inputs")
        return native.NativeExecutor(
            arrays, {**options, "dense_kernel": "blas"}, scope, threads
        )
    if name in ("scipy-lu-total", "scipy-lu-reuse"):
        from .adapters import scipy

        return scipy.SparseLuExecutor(
            arrays, options, scope, reuse=name.endswith("reuse")
        )
    if name in ("scipy-cg", "scipy-pcg"):
        from .adapters import scipy

        return scipy.ScipyKrylovExecutor(
            arrays, options, scope, jacobi=name == "scipy-pcg"
        )
    aliases = {
        "native-cg": ("native", "none"),
        "native-pcg": ("native", "jacobi"),
        "scipy-reference-cg": ("scipy-out", "none"),
        "scipy-reference-pcg": ("scipy-out", "jacobi"),
        "torch-cg": ("torch-eager-out", "none"),
        "torch-pcg": ("torch-eager-out", "jacobi"),
    }
    if name in aliases:
        name, preconditioner = aliases[name]
        options = {**options, "preconditioner": preconditioner}
    if name == "native":
        return native.NativeExecutor(
            arrays, {**options, "dense_kernel": "native"}, scope, threads
        )
    if name == "numpy-out":
        if hasattr(arrays[0], "indptr"):
            raise ValueError(
                "numpy-out requires dense inputs; use scipy-out for CSR"
            )
        return engine.ReferenceExecutor(numpy.NumpyOps(arrays), options, scope)
    if name == "scipy-out":
        return engine.ReferenceExecutor(
            scipy_ops.ScipyOps(arrays), options, scope
        )
    if name == "torch-eager-out":
        return engine.ReferenceExecutor(
            torch.TorchOps(arrays, threads), options, scope
        )
    raise ValueError(f"unknown implementation: {name}")
