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
    blas_enabled,
    build_info,
    openmp_enabled,
)
from _cgcore import solve_csr_into as _solve_csr_into
from _cgcore import solve_into as _solve_into

_MAX_SIZE_T = np.iinfo(np.uintp).max

__all__ = [
    "CSRMatrix",
    "blas_enabled",
    "build_info",
    "PreparedCsrSolver",
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
    """An owned solution with independently checked convergence information.

    Attributes
    ----------
    x : numpy.ndarray
        Owned float64 solution vector. The array remains writable.
    info : SolveInfo
        Read-only status, residual and operation counts from the native solve.

    Notes
    -----
    The dataclass fields cannot be rebound, but the contents of ``x`` can be
    modified. Check ``converged`` before using the solution.
    """

    x: np.ndarray
    info: SolveInfo

    @property
    def converged(self):
        """bool: Whether the true residual satisfied the requested tolerance."""
        return self.info.converged

    @property
    def status(self):
        """str: Native convergence, iteration-limit or numerical-failure status."""
        return self.info.status

    @property
    def iterations(self):
        """int: Number of completed solution updates."""
        return self.info.iterations

    @property
    def matvecs(self):
        """int: Matrix-vector products, including true-residual checks."""
        return self.info.matvecs

    @property
    def residual_norm(self):
        """float: Independently evaluated final Euclidean residual norm."""
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
    """Own a validated canonical compressed sparse row matrix.

    Parameters
    ----------
    indptr : array_like of int
        One-dimensional row pointers of length ``shape[0] + 1``. They must
        start at zero, be nondecreasing, and end at the number of values.
    indices : array_like of int
        One-dimensional nonnegative column indices, sorted and unique within
        each row. Each entry must be smaller than ``shape[1]``.
    data : array_like of float
        One-dimensional real values matching ``indices``. Values are copied
        into float64 storage.
    shape : tuple of int
        Nonnegative row and column counts. Solving additionally requires a
        nonempty square matrix.

    Raises
    ------
    ValueError
        Index arrays, dimensions, canonical ordering or value shape are invalid.
    TypeError
        Native dimensions or indices cannot be represented by ``size_t``.

    Notes
    -----
    Construction snapshots the arrays. Later changes to them do not alter the
    operator. Canonicalize SciPy CSR with ``sum_duplicates()`` and
    ``sort_indices()`` before wrapping arrays that need those operations.

    Examples
    --------
    >>> import cgpy
    >>> A = cgpy.CSRMatrix([0, 2, 4], [0, 1, 0, 1], [4., 1., 1., 3.], (2, 2))
    >>> A.shape, A.nnz
    ((2, 2), 4)
    """

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
        """tuple of int: Matrix dimensions ``(rows, columns)``."""
        return self._native.shape

    @property
    def nnz(self):
        """int: Number of stored entries, including explicitly stored zeros."""
        return self._native.nnz


def prepare(A, *, num_threads=1, preconditioner="none", dense_kernel="native"):
    """Snapshot a matrix and cache optional Jacobi setup for repeated solves.

    Parameters
    ----------
    A : array_like or CSRMatrix
        Nonempty square real SPD operator. Dense inputs are normalized to
        aligned C-contiguous float64 storage before the owning snapshot.
    num_threads : int, default=1
        Native CPU threads. Values greater than one require an OpenMP build.
        The BLAS kernel accepts only one thread.
    preconditioner : {'none', 'jacobi'}, default='none'
        Jacobi requires a finite strictly positive diagonal with a finite
        positive inverse and supports standard CG only.
    dense_kernel : {'native', 'blas'}, default='native'
        Explicit dense matrix-vector kernel. BLAS requires an enabled build,
        dense input and ``num_threads=1``. CSR requires ``'native'``.

    Returns
    -------
    PreparedSolver or PreparedCsrSolver
        Owning immutable matrix/preconditioner snapshot. Thread count and
        kernel choice are fixed; each solve selects its tolerances and fusion.

    Raises
    ------
    ValueError
        Dimensions, options, build capabilities or Jacobi setup are invalid.
    TypeError
        A native integer argument has an unsupported type or range.

    Notes
    -----
    Changes to the original matrix do not affect the prepared solver. Each
    overlapping solve needs a distinct output and ``create_workspace()``
    result. Preparation allocates; native solver vectors are reused afterward.

    Examples
    --------
    >>> import numpy as np
    >>> import cgpy
    >>> plan = cgpy.prepare([[4., 1.], [1., 3.]], preconditioner='jacobi')
    >>> x = np.zeros(2)
    >>> info = plan.solve_into(np.array([1., 2.]), x, workspace=plan.create_workspace())
    >>> info.converged
    True
    """
    if isinstance(A, CSRMatrix):
        return PreparedCsrSolver(A._native, num_threads, preconditioner, dense_kernel)
    return PreparedSolver(_matrix(A), num_threads, preconditioner, dense_kernel)


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
    dense_kernel="native",
):
    """Solve in place with strict arrays and caller-owned workspace.

    Parameters
    ----------
    A : numpy.ndarray or CSRMatrix
        Nonempty square real SPD operator. Dense arrays must already be
        aligned, C-contiguous and float64.
    b : numpy.ndarray
        Aligned C-contiguous float64 right-hand side of shape ``(n,)``.
    x : numpy.ndarray
        Writable aligned C-contiguous float64 vector of shape ``(n,)``.
        Supplies the initial guess and receives the current approximation.
    workspace : Workspace
        Reusable vectors with matching size, unused by overlapping calls.
    num_threads : int, default=1
        Positive native CPU thread count; multiple threads require OpenMP.
        BLAS requires one thread.
    rtol, atol : float, default=1e-8, 0.0
        Finite nonnegative tolerances. The true-residual threshold is
        ``max(atol, rtol * ||b||2)``.
    maxiter : int, default=1000
        Nonnegative update limit. Zero checks only the initial guess.
    algorithm : {'standard', 'explicit'}, default='standard'
        Residual recurrence or the explicit-residual legacy formulation.
    fused : bool, default=True
        Fuse supported CPU operations. Independent of algorithm and dense
        kernel; BLAS retains the selected vector update/reduction policy.
    residual_refresh : int, default=50
        Standard-CG true-residual replacement/restart interval. Zero disables
        periodic replacement; candidate and final checks remain enabled.
    preconditioner : {'none', 'jacobi'}, default='none'
        Construct Jacobi for this call; requires ``algorithm='standard'``.
    dense_kernel : {'native', 'blas'}, default='native'
        Explicit dense matrix-vector kernel. BLAS requires an enabled build,
        dense input and ``num_threads=1``; CSR requires ``'native'``.

    Returns
    -------
    SolveInfo
        Status and counts. Numerical breakdown or an iteration limit may
        leave an unconverged approximation in ``x``.

    Raises
    ------
    ValueError
        Shapes, dtype/layout, aliasing, workspace use, options or Jacobi setup
        violate the contract. Nonintegral/negative Python iteration limits
        also raise this exception.
    TypeError
        A native argument has an unsupported type or integer range.

    Notes
    -----
    There is no implicit array conversion. Keep ``x`` disjoint from ``A`` and
    ``b``. Borrowed arrays must remain alive and free from conflicting mutation
    until this synchronous call returns. Native computation releases the GIL.
    Use ``prepare`` to reuse matrix and Jacobi setup across calls.

    Examples
    --------
    >>> import numpy as np
    >>> import cgpy
    >>> A = np.array([[4., 1.], [1., 3.]])
    >>> x = np.zeros(2)
    >>> info = cgpy.solve_into(A, np.array([1., 2.]), x, workspace=cgpy.Workspace(2))
    >>> info.status
    'converged'
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
        dense_kernel=dense_kernel,
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
    dense_kernel="native",
):
    """Solve a real SPD system and return an owned solution.

    Parameters
    ----------
    A : array_like or CSRMatrix
        Nonempty square operator. Dense input is normalized to aligned
        C-contiguous float64 storage. Symmetry and positive definiteness are
        caller preconditions.
    b : array_like
        Real right-hand side of shape ``(n,)``, normalized to float64.
    x0 : array_like, optional
        Real initial guess of shape ``(n,)``. Omission selects zeros. The
        returned solution always owns a separate writable array.
    num_threads : int, default=1
        Positive native CPU thread count. Multiple threads require OpenMP;
        the BLAS kernel accepts only one thread.
    rtol, atol : float, default=1e-8, 0.0
        Finite nonnegative tolerances, with stopping threshold
        ``max(atol, rtol * ||b||2)``.
    maxiter : int, default=1000
        Nonnegative update limit. Zero checks only the initial guess.
    algorithm : {'standard', 'explicit'}, default='standard'
        Residual recurrence or the explicit-residual legacy formulation.
    fused : bool, default=True
        Fuse supported CPU operations. This is independent of algorithm and
        dense kernel. BLAS retains the selected vector fusion policy.
    residual_refresh : int, default=50
        Standard-CG true-residual replacement/restart interval. Zero disables
        periodic refresh, while candidate and final checks remain enabled.
    preconditioner : {'none', 'jacobi'}, default='none'
        Jacobi is prepared for this call and requires a finite positive
        diagonal/inverse and ``algorithm='standard'``.
    dense_kernel : {'native', 'blas'}, default='native'
        Explicit dense matrix-vector kernel. BLAS requires an enabled build,
        dense input and ``num_threads=1``. CSR requires ``'native'``.

    Returns
    -------
    SolveResult
        Owned solution and native status, true residual and operation counts.

    Raises
    ------
    ValueError
        Dimensions, real-valued conversion, options, build capabilities or
        Jacobi setup are invalid. Iteration limits must be nonnegative integers.
    TypeError
        A native argument has an unsupported type or integer range.

    Notes
    -----
    Inputs are preserved. Compatible ``A`` and ``b`` arrays can be borrowed
    until the synchronous call returns; do not modify them concurrently.
    Native computation releases the GIL. Numerical failure and iteration
    limits are reported in ``status``; always inspect ``converged``.

    Examples
    --------
    >>> import numpy as np
    >>> import cgpy
    >>> result = cgpy.solve([[4., 1.], [1., 3.]], [1., 2.], preconditioner='jacobi')
    >>> result.converged
    True
    >>> np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])
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
        dense_kernel=dense_kernel,
    )
    return SolveResult(x, info)
