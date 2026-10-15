"""Dense and sparse CG/PCG solvers for symmetric positive-definite systems."""

from ._core import blas_enabled, build_info, openmp_enabled
from .matrix import CsrMatrix
from .solver import (
    PreparedSolver,
    SolveInfo,
    SolveResult,
    Workspace,
    prepare,
    solve,
    solve_into,
)

__all__ = [
    "CsrMatrix",
    "PreparedSolver",
    "SolveInfo",
    "SolveResult",
    "Workspace",
    "blas_enabled",
    "build_info",
    "openmp_enabled",
    "prepare",
    "solve",
    "solve_into",
]
