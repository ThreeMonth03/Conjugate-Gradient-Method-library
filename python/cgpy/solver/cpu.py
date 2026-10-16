"""CPU entry points for dense and CSR conjugate gradient solves."""

import numpy as np
import numpy.typing as npt

from .. import _core
from .._interop import host
from ..matrix import csr
from . import _validation, result
from . import workspace as workspace_module


def prepare(
    A: npt.ArrayLike | csr.CsrMatrix,
    *,
    num_threads: int = 1,
    preconditioner: str = "none",
    dense_kernel: str = "native",
) -> _core.PreparedSolver:
    """Snapshot a matrix and cache optional Jacobi setup for repeated solves.

    Parameters
    ----------
    A : array_like or CsrMatrix
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
    PreparedSolver
        Owning immutable matrix/preconditioner snapshot. Thread count and
        kernel choice are fixed; each solve selects its tolerances and fusion.
        Create this type through ``prepare``; it has no public constructor.

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
    >>> info = plan.solve_into(
    ...     np.array([1., 2.]), x, workspace=plan.create_workspace()
    ... )
    >>> info.converged
    True
    """
    matrix = (
        A
        if isinstance(A, csr.CsrMatrix)
        else host.normalize_host_array(A, 2, name="A")
    )
    _validation.validate_operator_shape(matrix.shape)
    native_matrix = (
        matrix._native if isinstance(matrix, csr.CsrMatrix) else matrix
    )
    return _core.prepare(
        native_matrix,
        num_threads=num_threads,
        preconditioner=preconditioner,
        dense_kernel=dense_kernel,
    )


def solve_into(
    A: np.ndarray | csr.CsrMatrix,
    b: np.ndarray,
    x: np.ndarray,
    *,
    workspace: workspace_module.Workspace,
    num_threads: int = 1,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 1000,
    algorithm: str = "standard",
    fused: bool = True,
    residual_refresh: int = 50,
    preconditioner: str = "none",
    dense_kernel: str = "native",
) -> _core.SolveInfo:
    """Solve in place with strict arrays and caller-owned workspace.

    Parameters
    ----------
    A : numpy.ndarray or CsrMatrix
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
        Residual recurrence or the explicit-residual reference formulation.
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
    >>> info = cgpy.solve_into(
    ...     A, np.array([1., 2.]), x, workspace=cgpy.Workspace(2)
    ... )
    >>> info.status
    'converged'
    """
    native_A = A._native if isinstance(A, csr.CsrMatrix) else A
    return _core.solve_into(
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
    A: npt.ArrayLike | csr.CsrMatrix,
    b: npt.ArrayLike,
    x0: npt.ArrayLike | None = None,
    *,
    num_threads: int = 1,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 1000,
    algorithm: str = "standard",
    fused: bool = True,
    residual_refresh: int = 50,
    preconditioner: str = "none",
    dense_kernel: str = "native",
) -> result.SolveResult[np.ndarray]:
    """Solve a real SPD system and return an owned solution.

    Parameters
    ----------
    A : array_like or CsrMatrix
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
        Residual recurrence or the explicit-residual reference formulation.
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
        Jacobi setup are invalid. Iteration limits must be integers >= 0.
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
    >>> result = cgpy.solve(
    ...     [[4., 1.], [1., 3.]], [1., 2.], preconditioner='jacobi'
    ... )
    >>> result.converged
    True
    >>> np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])
    """
    maxiter, residual_refresh = _core._validate_iteration_limits(
        maxiter, residual_refresh
    )
    if not isinstance(A, csr.CsrMatrix):
        A = host.normalize_host_array(A, 2, name="A")
    _validation.validate_operator_shape(A.shape)
    b = host.normalize_host_array(b, 1, name="b")
    if b.size != A.shape[0]:
        raise ValueError("b must be a vector matching A")
    x = (
        np.zeros_like(b)
        if x0 is None
        else host.normalize_host_array(x0, 1, name="x0", copy=True)
    )
    if x.size != b.size:
        raise ValueError("x0 must be a vector matching b")
    info = solve_into(
        A,
        b,
        x,
        workspace=workspace_module.Workspace(b.size),
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
    return result.SolveResult(x, info)
