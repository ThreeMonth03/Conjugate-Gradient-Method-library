"""Explicit synchronous CG/PCG policies for device-resident systems."""

from .. import _core
from .._runtime import cuda as cuda_runtime
from .._runtime import validation
from ..matrix import csr
from ..matrix import cuda as matrix
from ..storage import cuda as storage
from . import cuda_workspace, result


def _native(value):
    return (
        value._native
        if isinstance(
            value, (storage.Vector, matrix.DenseMatrix, matrix.CSRMatrix)
        )
        else value
    )


class PreparedSolver:
    """Own an immutable GPU matrix and optional Jacobi inverse.

    Construct with ``prepare`` or ``prepare_host``. The device, explicit
    dense kernel and preconditioner are fixed; fusion, tolerances and refresh
    policy are selected on each solve. Matrix snapshots and Jacobi setup are
    outside prepared-call timing. Distinct workspace/output pairs are required
    for overlapping solves; input storage must stay free of conflicting writes.
    """

    def __init__(self) -> None:
        raise TypeError("use prepare or prepare_host")

    @property
    def size(self) -> int:
        """int: Nonempty square system dimension."""
        return self._native.size

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple: Square matrix dimensions."""
        return self.size, self.size

    @property
    def device(self) -> int:
        """int: Device ordinal of the matrix snapshot."""
        return self._native.device

    @property
    def dense_kernel(self) -> str:
        """str: Explicit native or cublas dense kernel; CSR requires native."""
        return self._native.dense_kernel

    @property
    def preconditioner(self) -> str:
        """str: Fixed none or jacobi selection."""
        return self._native.preconditioner

    @property
    def preconditioner_setup_ns(self) -> int:
        """int: Completed Jacobi setup nanoseconds, or zero without Jacobi."""
        return self._native.preconditioner_setup_ns

    @property
    def preconditioner_nbytes(self) -> int:
        """int: Stored Jacobi inverse bytes, excluding runtime resources."""
        return self._native.preconditioner_nbytes

    def create_workspace(self, *, stream: int = 0) -> cuda_workspace.Workspace:
        """Allocate reusable solver storage on the selected borrowed stream."""
        return cuda_workspace.Workspace(
            self.size,
            device=self.device,
            stream=stream,
            dense_kernel=self.dense_kernel,
        )

    def create_vector(self, *, stream: int = 0) -> storage.Vector:
        """Allocate an independent zero output on this plan's device."""
        return storage.Vector(self.size, device=self.device, stream=stream)

    def solve_into(
        self,
        b: object,
        x: object,
        *,
        workspace: cuda_workspace.Workspace,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 1000,
        algorithm: str = "standard",
        fused: bool = True,
        residual_refresh: int = 50,
    ) -> _core.SolveInfo:
        """Solve with cached setup and caller-owned GPU output/workspace.

        Parameters
        ----------
        b, x : Vector or CUDA Array Interface object
            Contiguous float64 device vectors. x is writable initial guess and
            output, disjoint from b. Reset x before a repeated comparison.
        workspace : Workspace
            Matching size/device, unused by overlapping calls. Its stream
            determines execution and the version-2 producer readiness contract.
        rtol, atol : float
            Finite nonnegative true-residual tolerances.
        maxiter : int
            Nonnegative update limit; zero checks only the initial guess.
        algorithm : {'standard', 'explicit'}
            Shared CPU/GPU policy. Jacobi requires standard CG.
        fused : bool
            Select fused or separate native operations for either kernel.
        residual_refresh : int
            Standard-CG residual replacement/restart interval; zero disables
            periodic replacement, retaining candidate/final true checks.

        Returns
        -------
        cgpy.SolveInfo
            Numerical status and counts. Numerical failure may leave a current
            approximation in x. Native solving releases the GIL and returns
            synchronously on the workspace stream.

        Raises
        ------
        ValueError
            Invalid array type/layout/device, aliasing, workspace, options or
            kernel/preconditioner combination.
        RuntimeError
            CUDA allocation, stream or execution failure.
        """
        maxiter, residual_refresh = validation.iteration_limits(
            maxiter, residual_refresh
        )
        return self._native.solve_into(
            _native(b),
            _native(x),
            workspace=workspace._native,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            algorithm=algorithm,
            fused=fused,
            residual_refresh=residual_refresh,
        )


def prepare(
    A: object,
    *,
    device: int = 0,
    stream: int = 0,
    dense_kernel: str = "native",
    preconditioner: str = "none",
) -> PreparedSolver:
    """Snapshot a device matrix and cache optional Jacobi setup.

    A must be a DenseMatrix, CSRMatrix or contiguous float64 CUDA Array
    Interface matrix. It must be nonempty, square and SPD.
    Kernel is explicitly native or cublas; CSR requires native. Jacobi requires
    a finite positive diagonal/inverse and standard CG. Version-2 producers
    must be ready on the borrowed stream. Setup is synchronous; future solves
    select their stream by creating a workspace. Returns PreparedSolver.
    """
    result = PreparedSolver.__new__(PreparedSolver)
    result._native = cuda_runtime.load().PreparedSolver(
        _native(A), device, stream, dense_kernel, preconditioner
    )
    return result


def prepare_host(
    A: object,
    *,
    device: int = 0,
    stream: int = 0,
    dense_kernel: str = "native",
    preconditioner: str = "none",
) -> PreparedSolver:
    """Upload a real host dense/CPU-CSR matrix and prepare a snapshot.

    Transfers, snapshot allocation and optional Jacobi setup are included in
    preparation. Repeated solve_into calls keep these costs outside timing.
    Kernel, device and preconditioner have the same contract as prepare.
    """
    source = (
        A._native
        if isinstance(A, csr.CSRMatrix)
        else validation.host_array(A, 2)
    )
    result = PreparedSolver.__new__(PreparedSolver)
    result._native = cuda_runtime.load().PreparedSolver.from_host(
        source, device, stream, dense_kernel, preconditioner
    )
    return result


def solve_into(
    A: object,
    b: object,
    x: object,
    *,
    workspace: cuda_workspace.Workspace,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 1000,
    algorithm: str = "standard",
    fused: bool = True,
    residual_refresh: int = 50,
    preconditioner: str = "none",
    dense_kernel: str = "native",
) -> _core.SolveInfo:
    """Solve a device-resident system with caller-owned output and workspace.

    A is an owned CUDA matrix or a contiguous float64 CUDA Array Interface
    dense matrix; b/x are device float64 vectors. No host conversion or CPU
    fallback occurs. Output and workspace must be disjoint from inputs and
    unused by overlapping calls. The workspace fixes the device and stream.
    Version-2 producers must be ready on that stream; version-3 dependencies
    are honored. Each call constructs Jacobi when selected. Use prepare to
    cache setup. Options, exceptions, synchronous completion and SolveInfo
    follow PreparedSolver.solve_into, including the true-residual stopping rule
    max(atol, rtol*||b||2). Callers must supply an SPD matrix.
    """
    maxiter, residual_refresh = validation.iteration_limits(
        maxiter, residual_refresh
    )
    return cuda_runtime.load().solve_into(
        _native(A),
        _native(b),
        _native(x),
        workspace=workspace._native,
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
    A: object,
    b: object,
    x0: object | None = None,
    *,
    device: int = 0,
    stream: int = 0,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 1000,
    algorithm: str = "standard",
    fused: bool = True,
    residual_refresh: int = 50,
    preconditioner: str = "none",
    dense_kernel: str = "native",
) -> result.SolveResult[storage.Vector]:
    """Solve a device system with fresh output, workspace and per-call setup.

    A is an owned CUDA matrix or contiguous float64 CUDA Array Interface dense
    matrix; b and optional x0 are device float64 vectors. Omitted x0 selects
    zero. Inputs are preserved; the result owns a separate GPU vector. Device
    and borrowed stream are explicit. No host upload/download is performed.
    Standard/explicit CG, Jacobi, fusion, tolerance and error contracts follow
    solve_into; numerical failures are reported in the returned status.
    Native computation releases the GIL and finishes before returning.

    Returns
    -------
    SolveResult
        GPU Vector and status/counts. Download with result.x.to_host().
    """
    if not hasattr(b, "__cuda_array_interface__"):
        raise ValueError(
            "b must be a device vector with __cuda_array_interface__"
        )
    shape = tuple(b.__cuda_array_interface__["shape"])
    if len(shape) != 1:
        raise ValueError("b must be a one-dimensional device vector")
    x = storage.Vector._empty(shape[0], device, stream)
    if x0 is None:
        x.fill(0.0, stream=stream)
    else:
        x.copy_from_device(x0, stream=stream)
    info = solve_into(
        A,
        b,
        x,
        workspace=cuda_workspace.Workspace(
            shape[0], device=device, stream=stream, dense_kernel=dense_kernel
        ),
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
