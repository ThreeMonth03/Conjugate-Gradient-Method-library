"""Types for the optional native CUDA boundary and its array capability."""

from typing import Never, SupportsIndex

import numpy as np
import numpy.typing as npt

from . import _core
from ._typing import (
    Algorithm,
    CudaDenseKernel,
    ExecutionMode,
    Preconditioner,
    SupportsCudaArrayInterface,
)

class Vector:
    def __init__(
        self, size: int, device: int = 0, stream: int = 0
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def device(self) -> int: ...
    @property
    def __cuda_array_interface__(self) -> dict[str, object]: ...
    def fill(self, value: float, stream: int = 0) -> None: ...
    def copy_from_host(
        self, source: npt.NDArray[np.float64], stream: int = 0
    ) -> None: ...
    def copy_from_device(
        self, source: SupportsCudaArrayInterface, stream: int = 0
    ) -> None: ...
    def to_host(self, stream: int = 0) -> npt.NDArray[np.float64]: ...

class DenseMatrix:
    # The native factories create snapshots; no Python constructor is bound.
    def __init__(self, _factory_only: Never, /) -> None: ...
    @staticmethod
    def from_host(
        source: npt.NDArray[np.float64], device: int = 0, stream: int = 0
    ) -> DenseMatrix: ...
    @staticmethod
    def from_device(
        source: SupportsCudaArrayInterface, device: int = 0, stream: int = 0
    ) -> DenseMatrix: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def device(self) -> int: ...

class CsrMatrix:
    # The native factories create snapshots; no Python constructor is bound.
    def __init__(self, _factory_only: Never, /) -> None: ...
    @staticmethod
    def from_host(
        source: _core.CsrMatrix, device: int = 0, stream: int = 0
    ) -> CsrMatrix: ...
    @staticmethod
    def from_device(
        indptr: SupportsCudaArrayInterface,
        indices: SupportsCudaArrayInterface,
        data: SupportsCudaArrayInterface,
        shape: tuple[int, int],
        device: int = 0,
        stream: int = 0,
    ) -> CsrMatrix: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def nnz(self) -> int: ...
    @property
    def device(self) -> int: ...

class Workspace:
    def __init__(
        self,
        size: int,
        device: int = 0,
        stream: int = 0,
        dense_kernel: CudaDenseKernel = "native",
        execution_mode: ExecutionMode = "host",
    ) -> None: ...
    @property
    def size(self) -> int: ...
    @property
    def device(self) -> int: ...
    @property
    def stream(self) -> int: ...
    @property
    def dense_kernel(self) -> CudaDenseKernel: ...
    @property
    def execution_mode(self) -> ExecutionMode: ...

class PreparedSolver:
    def __init__(
        self,
        matrix: DenseMatrix | CsrMatrix | SupportsCudaArrayInterface,
        device: int = 0,
        stream: int = 0,
        dense_kernel: CudaDenseKernel = "native",
        preconditioner: Preconditioner = "none",
    ) -> None: ...
    @staticmethod
    def from_host(
        matrix: npt.NDArray[np.float64] | _core.CsrMatrix,
        device: int = 0,
        stream: int = 0,
        dense_kernel: CudaDenseKernel = "native",
        preconditioner: Preconditioner = "none",
    ) -> PreparedSolver: ...
    @property
    def size(self) -> int: ...
    @property
    def device(self) -> int: ...
    @property
    def dense_kernel(self) -> CudaDenseKernel: ...
    @property
    def preconditioner(self) -> Preconditioner: ...
    @property
    def preconditioner_setup_ns(self) -> int: ...
    @property
    def preconditioner_nbytes(self) -> int: ...
    def solve_into(
        self,
        b: SupportsCudaArrayInterface,
        x: SupportsCudaArrayInterface,
        *,
        workspace: Workspace,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: SupportsIndex = 1000,
        algorithm: Algorithm = "standard",
        fused: bool = True,
        residual_refresh: SupportsIndex = 50,
    ) -> _core.SolveInfo: ...

def solve_into(
    A: DenseMatrix | CsrMatrix | SupportsCudaArrayInterface,
    b: SupportsCudaArrayInterface,
    x: SupportsCudaArrayInterface,
    *,
    workspace: Workspace,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: SupportsIndex = 1000,
    algorithm: Algorithm = "standard",
    fused: bool = True,
    residual_refresh: SupportsIndex = 50,
    preconditioner: Preconditioner = "none",
) -> _core.SolveInfo: ...
def _initial_output(
    b: SupportsCudaArrayInterface,
    x0: SupportsCudaArrayInterface | None,
    device: int,
    stream: int,
) -> Vector: ...
def is_available(device: int = 0) -> bool: ...
def synchronize(device: int = 0, stream: int = 0) -> None: ...
def runtime_info(device: int = 0) -> dict[str, object]: ...
def build_info() -> dict[str, object]: ...
