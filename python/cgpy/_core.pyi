"""Types for the strict native CPU boundary."""

from typing import Any, Never, SupportsIndex

import numpy as np
import numpy.typing as npt

from ._typing import (
    Algorithm,
    CpuDenseKernel,
    Preconditioner,
    SolveStatus,
)

class CsrMatrix:
    def __init__(
        self,
        indptr: npt.NDArray[np.integer[Any]],
        indices: npt.NDArray[np.integer[Any]],
        data: npt.NDArray[np.float64],
        shape: tuple[int, int],
    ) -> None: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...

class SolveInfo:
    # Native calls create this record; it has no Python constructor.
    def __init__(self, _factory_only: Never, /) -> None: ...
    @property
    def status(self) -> SolveStatus: ...
    @property
    def converged(self) -> bool: ...
    @property
    def iterations(self) -> int: ...
    @property
    def matvecs(self) -> int: ...
    @property
    def residual_norm(self) -> float: ...
    @property
    def restarts(self) -> int: ...

class Workspace:
    def __init__(self, size: int) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def capacity(self) -> int: ...

class PreparedSolver:
    # prepare() creates this snapshot; it has no Python constructor.
    def __init__(self, _factory_only: Never, /) -> None: ...
    def create_workspace(self) -> Workspace: ...
    @property
    def size(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def preconditioner_nbytes(self) -> int: ...
    @property
    def num_threads(self) -> int: ...
    @property
    def dense_kernel(self) -> CpuDenseKernel: ...
    @property
    def preconditioner_setup_ns(self) -> int: ...
    @property
    def preconditioner(self) -> Preconditioner: ...
    def solve_into(
        self,
        b: npt.NDArray[np.float64],
        x: npt.NDArray[np.float64],
        *,
        workspace: Workspace,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: SupportsIndex = 1000,
        algorithm: Algorithm = "standard",
        fused: bool = True,
        residual_refresh: SupportsIndex = 50,
    ) -> SolveInfo: ...

def prepare(
    A: npt.NDArray[np.float64] | CsrMatrix,
    *,
    num_threads: int = 1,
    preconditioner: Preconditioner = "none",
    dense_kernel: CpuDenseKernel = "native",
) -> PreparedSolver: ...
def solve_into(
    A: npt.NDArray[np.float64] | CsrMatrix,
    b: npt.NDArray[np.float64],
    x: npt.NDArray[np.float64],
    *,
    workspace: Workspace,
    num_threads: int = 1,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: SupportsIndex = 1000,
    algorithm: Algorithm = "standard",
    fused: bool = True,
    residual_refresh: SupportsIndex = 50,
    preconditioner: Preconditioner = "none",
    dense_kernel: CpuDenseKernel = "native",
) -> SolveInfo: ...
def _validate_iteration_limits(
    maxiter: SupportsIndex, residual_refresh: SupportsIndex
) -> tuple[int, int]: ...
def openmp_enabled() -> bool: ...
def blas_enabled() -> bool: ...
def build_info() -> dict[str, object]: ...
