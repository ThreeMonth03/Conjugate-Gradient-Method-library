"""Shared static declarations for solver options and CUDA array inputs."""

from typing import Literal, Protocol


class SupportsCudaArrayInterface(Protocol):
    """Expose a CUDA descriptor whose metadata is validated at runtime."""

    @property
    def __cuda_array_interface__(self) -> object: ...


type Algorithm = Literal["standard", "explicit"]
type Preconditioner = Literal["none", "jacobi"]
type CpuDenseKernel = Literal["native", "blas"]
type CudaDenseKernel = Literal["native", "cublas"]
type ExecutionMode = Literal["host", "device"]
type SolveStatus = Literal[
    "converged", "max_iterations", "non_positive_curvature", "non_finite"
]
