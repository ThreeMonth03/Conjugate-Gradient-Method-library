"""Explicit CUDA CG/PCG with device-owned solutions and synchronous calls.

Importing this module works with CPU-only builds. CUDA operations require an
optional extension, its NVIDIA runtime dependencies and a compatible GPU.
Uploads use explicitly named ``from_host`` / ``prepare_host`` methods.
"""

from .._runtime.cuda import (
    build_info,
    enabled,
    is_available,
    runtime_info,
    synchronize,
)
from .._typing import SupportsCudaArrayInterface
from ..matrix.cuda import CsrMatrix, DenseMatrix
from ..solver.cuda import (
    PreparedSolver,
    Session,
    Workspace,
    prepare,
    prepare_host,
    solve,
    solve_into,
)
from ..solver.result import SolveResult
from ..storage.cuda import Vector

__all__ = [
    "CsrMatrix",
    "DenseMatrix",
    "PreparedSolver",
    "Session",
    "SolveResult",
    "SupportsCudaArrayInterface",
    "Vector",
    "Workspace",
    "build_info",
    "enabled",
    "is_available",
    "prepare",
    "prepare_host",
    "runtime_info",
    "solve",
    "solve_into",
    "synchronize",
]
