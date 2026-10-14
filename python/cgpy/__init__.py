"""CPU conjugate gradient solvers for symmetric positive-definite systems."""

from dataclasses import dataclass
from operator import index

import numpy as np
from _cgcore import CSRMatrix as _CSRMatrix
from _cgcore import (
    PreparedCsrSolver,
    PreparedSolver,
    SolveInfo,
    Workspace,
    build_info,
    openmp_enabled,
)
from _cgcore import solve_csr_into as _solve_csr_into
from _cgcore import solve_into as _solve_into

_MAX_SIZE_T = np.iinfo(np.uintp).max

__all__ = [
    "CSRMatrix",
    "build_info",
    "PreparedSolver",
    "SolveInfo",
    "SolveResult",
    "Workspace",
    "openmp_enabled",
    "prepare",
    "solve",
    "solve_into",
]


@dataclass(frozen=True)
class SolveResult:
    """Own the solution and report the final independently checked residual."""

    x: np.ndarray
    info: SolveInfo

    @property
    def converged(self):
        return self.info.converged

    @property
    def status(self):
        return self.info.status

    @property
    def iterations(self):
        return self.info.iterations

    @property
    def matvecs(self):
        return self.info.matvecs

    @property
    def residual_norm(self):
        return self.info.residual_norm


def _real_array(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must contain real values; complex inputs are unsupported")
    return array


def _matrix(A):
    A = np.require(_real_array(A, "A"), dtype=np.float64, requirements=("C", "A"))
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a nonempty square matrix")
    return A


def _iteration_limits(maxiter, residual_refresh):
    counts = []
    for name, value in (("maxiter", maxiter), ("residual_refresh", residual_refresh)):
        try:
            count = index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a nonnegative integer") from exc
        if count < 0 or count > _MAX_SIZE_T:
            raise ValueError(f"{name} must fit a nonnegative size_t")
        counts.append(count)
    return counts


class CSRMatrix:
    """Own canonical CSR arrays. Rows must have sorted, unique column indices."""

    def __init__(self, indptr, indices, data, shape):
        rowptr, columns = np.asarray(indptr), np.asarray(indices)
        if rowptr.ndim != 1 or columns.ndim != 1:
            raise ValueError("CSR index arrays must be one-dimensional")
        if rowptr.dtype.kind not in "iu" or (columns.size and columns.dtype.kind not in "iu"):
            raise ValueError("CSR index arrays must contain integers")
        if np.any(rowptr < 0) or np.any(columns < 0):
            raise ValueError("CSR indices must be nonnegative")
        values = _real_array(data, "data")
        if values.ndim != 1:
            raise ValueError("CSR data must be one-dimensional")
        self._native = _CSRMatrix(rowptr, columns, values, shape)

    @property
    def shape(self):
        return self._native.shape

    @property
    def nnz(self):
        return self._native.nnz


def prepare(A, *, num_threads=1, preconditioner="none"):
    """Snapshot A for repeated solves; each active solve needs its own workspace."""
    if isinstance(A, CSRMatrix):
        return PreparedCsrSolver(A._native, num_threads, preconditioner)
    return PreparedSolver(_matrix(A), num_threads, preconditioner)


def solve_into(
    A,
    b,
    x,
    *,
    workspace,
    num_threads=1,
    rtol=1e-8,
    atol=0.0,
    maxiter=1000,
    algorithm="standard",
    fused=True,
    residual_refresh=50,
    preconditioner="none",
):
    """Solve in place, treating x as the initial guess. No implicit array conversion.

    A may be an owned CSRMatrix or an aligned C-contiguous float64 NumPy array.
    b and x must be aligned C-contiguous float64 NumPy arrays. x must be writable
    and disjoint from A and b. Borrowed arrays must not be mutated concurrently
    while the GIL is released.
    The workspace must match the system size and cannot serve concurrent calls.
    """
    maxiter, residual_refresh = _iteration_limits(maxiter, residual_refresh)
    implementation = _solve_csr_into if isinstance(A, CSRMatrix) else _solve_into
    native_A = A._native if isinstance(A, CSRMatrix) else A
    return implementation(
        native_A,
        b,
        x,
        workspace=workspace,
        num_threads=num_threads,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        algorithm=algorithm,
        fused=fused,
        residual_refresh=residual_refresh,
        preconditioner=preconditioner,
    )


def solve(
    A,
    b,
    x0=None,
    *,
    num_threads=1,
    rtol=1e-8,
    atol=0.0,
    maxiter=1000,
    algorithm="standard",
    fused=True,
    residual_refresh=50,
    preconditioner="none",
):
    """Return a new solution; preserve inputs and normalize their dtype/layout.

    SPD is a caller precondition. Invalid array dimensions/options raise
    ValueError; numerical failure or an iteration limit is returned in status.
    Compatible A and b arrays are borrowed only for this synchronous call.
    """
    maxiter, residual_refresh = _iteration_limits(maxiter, residual_refresh)
    if isinstance(A, CSRMatrix):
        if A.shape[0] == 0 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a nonempty square matrix")
    else:
        A = _matrix(A)
    b = np.require(_real_array(b, "b"), dtype=np.float64, requirements=("C", "A"))
    if b.ndim != 1 or b.size != A.shape[0]:
        raise ValueError("b must be a vector matching A")
    x = (
        np.zeros_like(b)
        if x0 is None
        else np.array(_real_array(x0, "x0"), dtype=np.float64, order="C", copy=True)
    )
    if x.ndim != 1 or x.size != b.size:
        raise ValueError("x0 must be a vector matching b")
    info = solve_into(
        A,
        b,
        x,
        workspace=Workspace(b.size),
        num_threads=num_threads,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        algorithm=algorithm,
        fused=fused,
        residual_refresh=residual_refresh,
        preconditioner=preconditioner,
    )
    return SolveResult(x, info)
