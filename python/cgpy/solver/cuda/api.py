"""Explicit CUDA preparation and one-shot solver entry points."""

from ... import _core
from ..._interop import cuda as cuda_interop
from ..._interop import host
from ..._runtime import cuda as cuda_runtime
from ...matrix import csr
from ...storage import cuda as storage
from .. import result
from . import prepared
from . import workspace as workspace_module


def prepare(
    A: object,
    *,
    device: int = 0,
    stream: int = 0,
    dense_kernel: str = "native",
    preconditioner: str = "none",
) -> prepared.PreparedSolver:
    """Snapshot a device matrix and cache optional Jacobi setup.

    A must be a DenseMatrix, CsrMatrix or contiguous float64 CUDA Array
    Interface matrix. It must be nonempty, square and SPD.
    Kernel is explicitly native or cublas; CSR requires native. Jacobi requires
    a finite positive diagonal/inverse and standard CG. Version-2 producers
    must be ready on the borrowed stream. Setup is synchronous; future solves
    select their stream by creating a workspace. Returns PreparedSolver.
    """
    result = prepared.PreparedSolver.__new__(prepared.PreparedSolver)
    result._native = cuda_runtime.load().PreparedSolver(
        cuda_interop.native_object(A),
        device,
        stream,
        dense_kernel,
        preconditioner,
    )
    return result


def prepare_host(
    A: object,
    *,
    device: int = 0,
    stream: int = 0,
    dense_kernel: str = "native",
    preconditioner: str = "none",
) -> prepared.PreparedSolver:
    """Upload a real host dense/CPU-CSR matrix and prepare a snapshot.

    Transfers, snapshot allocation and optional Jacobi setup are included in
    preparation. Repeated solve_into calls keep these costs outside timing.
    Kernel, device and preconditioner have the same contract as prepare.
    """
    source = (
        A._native
        if isinstance(A, csr.CsrMatrix)
        else host.normalize_host_array(A, 2)
    )
    result = prepared.PreparedSolver.__new__(prepared.PreparedSolver)
    result._native = cuda_runtime.load().PreparedSolver.from_host(
        source, device, stream, dense_kernel, preconditioner
    )
    return result


def solve_into(
    A: object,
    b: object,
    x: object,
    *,
    workspace: workspace_module.Workspace,
    rtol: float = 1e-8,
    atol: float = 0.0,
    maxiter: int = 1000,
    algorithm: str = "standard",
    fused: bool = True,
    residual_refresh: int = 50,
    preconditioner: str = "none",
) -> _core.SolveInfo:
    """Solve a device-resident system with caller-owned output and workspace.

    A is an owned CUDA matrix or a contiguous float64 CUDA Array Interface
    dense matrix; b/x are device float64 vectors. No host conversion or CPU
    fallback occurs. Output and workspace must be disjoint from inputs and
    unused by overlapping calls. The workspace fixes the device, stream,
    dense kernel and host/device scalar-control execution_mode.
    Version-2 producers must be ready on that stream; version-3 dependencies
    are honored. Each call constructs Jacobi when selected. Use prepare to
    cache setup. Options, exceptions, synchronous completion and SolveInfo
    follow PreparedSolver.solve_into, including the true-residual stopping rule
    max(atol, rtol*||b||2). Callers must supply an SPD matrix.
    """
    return cuda_runtime.load().solve_into(
        cuda_interop.native_object(A),
        cuda_interop.native_object(b),
        cuda_interop.native_object(x),
        workspace=workspace._native,
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        algorithm=algorithm,
        fused=fused,
        residual_refresh=residual_refresh,
        preconditioner=preconditioner,
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
    execution_mode: str = "host",
) -> result.SolveResult[storage.Vector]:
    """Solve a device system with fresh output, workspace and per-call setup.

    A is an owned CUDA matrix or contiguous float64 CUDA Array Interface dense
    matrix; b and optional x0 are device float64 vectors. Omitted x0 selects
    zero. Inputs are preserved; the result owns a separate GPU vector. Device
    and borrowed stream are explicit. No host upload/download is performed.
    Standard/explicit CG, Jacobi, fusion, tolerance and error contracts follow
    solve_into; numerical failures are reported in the returned status.
    Native computation releases the GIL and finishes before returning.
    execution_mode is explicitly host (default) or device, selecting the new
    workspace's scalar-control engine independently of dense_kernel/fused.
    Device mode retains recurrence scalars on the GPU; completed status packets
    and true-residual checks still synchronize with the host.

    Returns
    -------
    SolveResult
        GPU Vector and status/counts. Download with result.x.to_host().
    """
    maxiter, residual_refresh = _core._validate_iteration_limits(
        maxiter, residual_refresh
    )
    x, size = _initial_output(b, x0, device, stream)
    info = solve_into(
        A,
        b,
        x,
        workspace=workspace_module.Workspace(
            size,
            device=device,
            stream=stream,
            dense_kernel=dense_kernel,
            execution_mode=execution_mode,
        ),
        rtol=rtol,
        atol=atol,
        maxiter=maxiter,
        algorithm=algorithm,
        fused=fused,
        residual_refresh=residual_refresh,
        preconditioner=preconditioner,
    )
    return result.SolveResult(x, info)


def _initial_output(b, x0, device, stream):
    """Allocate/reset fresh output without retaining caller inputs."""
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
    return x, shape[0]
