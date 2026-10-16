"""Shared static declarations for existing solver option strings."""

from typing import Literal

type Algorithm = Literal["standard", "explicit"]
type Preconditioner = Literal["none", "jacobi"]
type CpuDenseKernel = Literal["native", "blas"]
type CudaDenseKernel = Literal["native", "cublas"]
type ExecutionMode = Literal["host", "device"]
type SolveStatus = Literal[
    "converged", "max_iterations", "non_positive_curvature", "non_finite"
]
