"""Explicit CUDA float64 CG/PCG with synchronous calls and device-owned results.

Importing this module works with CPU-only builds. CUDA operations require an
enabled extension, the declared NVIDIA runtime packages and a compatible GPU.
Host uploads use explicitly named ``from_host`` / ``prepare_host`` methods.
Device calls never select a CPU fallback or change the current PyTorch stream.
"""

import ctypes
import importlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import CSRMatrix as HostCSRMatrix
from . import SolveInfo, _iteration_limits, _real_array

__all__ = [
    "CSRMatrix",
    "DenseMatrix",
    "PreparedSolver",
    "SolveResult",
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
_extension = None
_libraries = []


def _load():
    global _extension
    if _extension is not None:
        return _extension
    if importlib.util.find_spec("_cgcuda") is None:
        raise RuntimeError("CUDA requires a cgpy build with CG_ENABLE_CUDA=ON")
    # These are public NVIDIA distributions, independent of PyTorch's layout.
    for package, directory, names in (
        ("nvidia-cuda-runtime-cu12", "cuda_runtime", ("libcudart.so.12",)),
        ("nvidia-cublas-cu12", "cublas", ("libcublasLt.so.12", "libcublas.so.12")),
    ):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            continue  # A developer's explicitly configured loader can supply CUDA.
        for name in names:
            path = Path(distribution.locate_file(f"nvidia/{directory}/lib/{name}"))
            if path.is_file():
                try:
                    _libraries.append(ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL))
                except OSError as error:
                    raise RuntimeError(f"CUDA runtime library could not load: {name}") from error
    try:
        _extension = importlib.import_module("_cgcuda")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "CUDA extension could not load; install the cgpy[cuda] runtimes"
        ) from error
    return _extension


def enabled():
    """Return whether the optional CUDA extension can load, without requiring a GPU."""
    try:
        _load()
    except (RuntimeError, OSError):
        return False
    return True


def is_available(device=0):
    """Return whether the extension loads and CUDA exposes the requested GPU.

    A disabled extension or absent/incompatible driver returns False. This
    capability query never selects a different backend or device. The GPU must
    also support an architecture included in the explicitly configured build.
    """
    return enabled() and bool(_load().is_available(device))


def build_info():
    """Return imported CUDA compiler/toolkit/architecture metadata or a disabled reason."""
    try:
        return dict(_load().build_info())
    except (RuntimeError, OSError) as error:
        return {"enabled": False, "reason": str(error)}


def runtime_info(device=0):
    """Describe a GPU's name, UUID, memory, compute capability and CUDA versions.

    ``driver_api_version`` is the driver's supported CUDA API version, rather
    than NVIDIA's display-driver release number. Raises RuntimeError when CUDA
    or the requested device is unavailable.
    """
    return dict(_load().runtime_info(device))


def synchronize(device=0, stream=0):
    """Wait for the selected CUDA stream, restoring the caller's current device.

    Parameters
    ----------
    device : int, default=0
        Explicit device ordinal.
    stream : int, default=0
        Borrowed CUDA stream handle on that device; zero is the legacy default.
        The caller must keep a nondefault stream alive until the call returns.
    """
    _load().synchronize(device, stream)


def _native(value):
    return value._native if isinstance(value, (Vector, DenseMatrix, CSRMatrix)) else value


def _array(value, ndim):
    result = np.require(_real_array(value, "source"), dtype=np.float64, requirements=("C", "A"))
    if result.ndim != ndim:
        raise ValueError(f"source must have {ndim} dimensions")
    return result


class Vector:
    """Own a writable float64 GPU vector, initialized to zero.

    Parameters
    ----------
    size : int
        Nonnegative number of elements.
    device : int, default=0
        Device ordinal, fixed for the allocation's lifetime.
    stream : int, default=0
        Borrowed stream for initialization. All methods finish their work
        before returning; CUDA Array Interface exports have no pending work.

    Notes
    -----
    ``__cuda_array_interface__`` permits zero-copy interoperability. Its
    consumer must retain this owner and avoid conflicting mutation. Inputs
    from PyTorch must be detached float64 contiguous CUDA tensors. Allocation
    and host transfers belong outside a prepared-solve timing interval.
    """

    def __init__(self, size, *, device=0, stream=0):
        self._native = _load().Vector(size, device, stream)
        self.fill(0.0, stream=stream)

    @classmethod
    def _empty(cls, size, device, stream):
        result = cls.__new__(cls)
        result._native = _load().Vector(size, device, stream)
        return result

    @classmethod
    def from_host(cls, source, *, device=0, stream=0):
        """Snapshot a real host vector, normalizing and uploading it once."""
        source = _array(source, 1)
        result = cls._empty(source.size, device, stream)
        result.copy_from_host(source, stream=stream)
        return result

    @property
    def size(self):
        """int: Number of allocated float64 elements."""
        return self._native.size

    @property
    def shape(self):
        """tuple: One-dimensional shape."""
        return (self.size,)

    @property
    def device(self):
        """int: Device ordinal of the owned allocation."""
        return self._native.device

    @property
    def __cuda_array_interface__(self):
        """dict: Version-3 device view; all producer work is already complete."""
        return self._native.__cuda_array_interface__

    def fill(self, value, *, stream=0):
        """Fill the vector on a borrowed stream and wait for completion."""
        self._native.fill(value, stream)

    def copy_from_host(self, source, *, stream=0):
        """Normalize and upload a matching host vector; return after completion."""
        self._native.copy_from_host(_array(source, 1), stream)

    def copy_from_device(self, source, *, stream=0):
        """Copy a matching contiguous float64 device vector on the specified stream.

        CUDA Array Interface version 3 producer dependencies are honored. A
        version-2 producer, including PyTorch, must be ready on this same
        stream or explicitly synchronized by the caller before the copy.
        """
        self._native.copy_from_device(_native(source), stream)

    def to_host(self, *, stream=0):
        """Download an independent NumPy float64 snapshot and wait for completion."""
        return self._native.to_host(stream)


class DenseMatrix:
    """Own an immutable dense float64 device snapshot.

    Construct with ``from_host`` or ``from_device``. The matrix has no public
    mutable storage view; prepared solvers take a further owning snapshot.
    """

    def __init__(self):
        raise TypeError("use DenseMatrix.from_host or DenseMatrix.from_device")

    @classmethod
    def from_host(cls, source, *, device=0, stream=0):
        """Normalize a real 2D host array and explicitly upload a snapshot."""
        result = cls.__new__(cls)
        result._native = _load().DenseMatrix.from_host(_array(source, 2), device, stream)
        return result

    @classmethod
    def from_device(cls, source, *, device=0, stream=0):
        """Snapshot a C-contiguous float64 CUDA Array Interface matrix.

        Version-2 producers must be ready on the specified stream. Version-3
        producer stream dependencies are honored. The call is synchronous.
        """
        result = cls.__new__(cls)
        result._native = _load().DenseMatrix.from_device(source, device, stream)
        return result

    @property
    def shape(self):
        """tuple: Matrix dimensions (rows, columns)."""
        return self._native.shape

    @property
    def device(self):
        """int: Device ordinal of the snapshot."""
        return self._native.device


class CSRMatrix:
    """Own an immutable canonical CSR device snapshot.

    Columns must be sorted, unique and in range in each row; pointers must be
    monotone, start at zero and end at nnz. Setup validates the indices once;
    it does not silently sort or coalesce them. Use native CUDA CSR kernels.
    """

    def __init__(self):
        raise TypeError("use CSRMatrix.from_host or CSRMatrix.from_device")

    @classmethod
    def from_host(cls, source, *, device=0, stream=0):
        """Explicitly upload a public CPU cgpy.CSRMatrix snapshot."""
        if not isinstance(source, HostCSRMatrix):
            raise TypeError("source must be cgpy.CSRMatrix")
        result = cls.__new__(cls)
        result._native = _load().CSRMatrix.from_host(source._native, device, stream)
        return result

    @classmethod
    def from_device(cls, indptr, indices, data, shape, *, device=0, stream=0):
        """Snapshot device CSR arrays after validating their canonical structure.

        Pointers and columns must use matching signed int32 or int64 types.
        Values must be float64. All three arrays must be contiguous on the
        selected device. A setup-time index download is included in this call.
        """
        result = cls.__new__(cls)
        result._native = _load().CSRMatrix.from_device(indptr, indices, data, shape, device, stream)
        return result

    @property
    def shape(self):
        """tuple: Matrix dimensions (rows, columns)."""
        return self._native.shape

    @property
    def nnz(self):
        """int: Number of stored entries."""
        return self._native.nnz

    @property
    def device(self):
        """int: Device ordinal of the snapshot."""
        return self._native.device


class Workspace:
    """Own five solver vectors, reduction scratch and a cuBLAS handle.

    Parameters
    ----------
    size : int
        Matrix dimension. Storage is allocated once and reused across solves.
    device : int, default=0
        Device ordinal, fixed at construction.
    stream : int, default=0
        Borrowed stream handle, fixed at construction; it must outlive this
        workspace. One workspace/output pair belongs to one active solve.

    Notes
    -----
    Distinct workspaces permit overlapping calls against an immutable plan.
    Each workspace owns its handle and reduction storage. Reuse on the same
    stream is safe because public solves return only after completion.
    """

    def __init__(self, size, *, device=0, stream=0):
        self._native = _load().Workspace(size, device, stream)

    @property
    def size(self):
        """int: Fixed system dimension."""
        return self._native.size

    @property
    def device(self):
        """int: Fixed device ordinal."""
        return self._native.device

    @property
    def stream(self):
        """int: Borrowed stream handle used by all workspace operations."""
        return self._native.stream


class PreparedSolver:
    """Own an immutable GPU matrix and optional Jacobi inverse.

    Construct with ``prepare`` or ``prepare_host``. The device, explicit
    dense kernel and preconditioner are fixed; fusion, tolerances and refresh
    policy are selected on each solve. Matrix snapshots and Jacobi setup are
    outside prepared-call timing. Distinct workspace/output pairs are required
    for overlapping solves; input storage must stay free of conflicting writes.
    """

    def __init__(self):
        raise TypeError("use prepare or prepare_host")

    @property
    def size(self):
        """int: Nonempty square system dimension."""
        return self._native.size

    @property
    def shape(self):
        """tuple: Square matrix dimensions."""
        return self.size, self.size

    @property
    def device(self):
        """int: Device ordinal of the matrix snapshot."""
        return self._native.device

    @property
    def dense_kernel(self):
        """str: Explicit native or cublas dense kernel; CSR requires native."""
        return self._native.dense_kernel

    @property
    def preconditioner(self):
        """str: Fixed none or jacobi selection."""
        return self._native.preconditioner

    @property
    def preconditioner_setup_ns(self):
        """int: Completed Jacobi setup nanoseconds, or zero without Jacobi."""
        return self._native.preconditioner_setup_ns

    @property
    def preconditioner_nbytes(self):
        """int: Stored Jacobi inverse bytes, excluding runtime/allocator resources."""
        return self._native.preconditioner_nbytes

    def create_workspace(self, *, stream=0):
        """Allocate independent reusable solver storage on the selected borrowed stream."""
        return Workspace(self.size, device=self.device, stream=stream)

    def create_vector(self, *, stream=0):
        """Allocate an independent zero-initialized output on this plan's device."""
        return Vector(self.size, device=self.device, stream=stream)

    def solve_into(
        self,
        b,
        x,
        *,
        workspace,
        rtol=1e-8,
        atol=0.0,
        maxiter=1000,
        algorithm="standard",
        fused=True,
        residual_refresh=50,
    ):
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
            Explicit fused or separate native operations, independent of kernel.
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
        maxiter, residual_refresh = _iteration_limits(maxiter, residual_refresh)
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


def prepare(A, *, device=0, stream=0, dense_kernel="native", preconditioner="none"):
    """Snapshot a device matrix and cache optional Jacobi setup.

    A must be a DenseMatrix, CSRMatrix or contiguous float64 CUDA Array
    Interface matrix. It must be nonempty, square and SPD (caller precondition).
    Kernel is explicitly native or cublas; CSR requires native. Jacobi requires
    a finite positive diagonal/inverse and standard CG. Version-2 producers
    must be ready on the borrowed stream. Setup is synchronous; future solves
    select their stream by creating a workspace. Returns PreparedSolver.
    """
    result = PreparedSolver.__new__(PreparedSolver)
    result._native = _load().PreparedSolver(
        _native(A), device, stream, dense_kernel, preconditioner
    )
    return result


def prepare_host(A, *, device=0, stream=0, dense_kernel="native", preconditioner="none"):
    """Explicitly upload a real host dense/CPU-CSR matrix and prepare a snapshot.

    Transfers, snapshot allocation and optional Jacobi setup are included in
    preparation. Repeated solve_into calls keep these costs outside timing.
    Kernel, device and preconditioner have the same contract as prepare.
    """
    source = A._native if isinstance(A, HostCSRMatrix) else _array(A, 2)
    result = PreparedSolver.__new__(PreparedSolver)
    result._native = _load().PreparedSolver.from_host(
        source, device, stream, dense_kernel, preconditioner
    )
    return result


def solve_into(
    A,
    b,
    x,
    *,
    workspace,
    rtol=1e-8,
    atol=0.0,
    maxiter=1000,
    algorithm="standard",
    fused=True,
    residual_refresh=50,
    preconditioner="none",
    dense_kernel="native",
):
    """Solve a device-resident system with caller-owned output and workspace.

    A is an owned CUDA matrix or a contiguous float64 CUDA Array Interface
    dense matrix; b/x are device float64 vectors. No host conversion or CPU
    fallback occurs. Output and workspace must be disjoint from inputs and
    unused by overlapping calls. The workspace fixes device and borrowed stream.
    Version-2 producers must be ready on that stream; version-3 dependencies
    are honored. Each call constructs Jacobi when selected. Use prepare to cache
    setup. Options, exceptions, synchronous completion and returned SolveInfo
    follow PreparedSolver.solve_into, including the true-residual stopping rule
    max(atol, rtol*||b||2). Symmetry and positive definiteness are caller duties.
    """
    maxiter, residual_refresh = _iteration_limits(maxiter, residual_refresh)
    return _load().solve_into(
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


@dataclass(frozen=True)
class SolveResult:
    """An owned GPU solution Vector and read-only native SolveInfo.

    Use x.to_host() for an explicitly timed or untimed download. Dataclass
    fields cannot be rebound; GPU vector contents remain writable. Always
    check converged before using the approximation.
    """

    x: Vector
    info: SolveInfo

    @property
    def converged(self):
        """bool: True-residual tolerance was satisfied."""
        return self.info.converged

    @property
    def status(self):
        """str: Convergence, update-limit or numerical-failure status."""
        return self.info.status

    @property
    def iterations(self):
        """int: Completed updates."""
        return self.info.iterations

    @property
    def matvecs(self):
        """int: Matrix-vector products including true-residual checks."""
        return self.info.matvecs

    @property
    def residual_norm(self):
        """float: Independently recomputed final residual norm."""
        return self.info.residual_norm


def solve(
    A,
    b,
    x0=None,
    *,
    device=0,
    stream=0,
    rtol=1e-8,
    atol=0.0,
    maxiter=1000,
    algorithm="standard",
    fused=True,
    residual_refresh=50,
    preconditioner="none",
    dense_kernel="native",
):
    """Solve a device-resident system with fresh output/workspace and per-call setup.

    A is an owned CUDA matrix or contiguous float64 CUDA Array Interface dense
    matrix; b and optional x0 are device float64 vectors. Omitted x0 selects
    zero. Inputs are preserved; the result owns a separate GPU vector. Device
    and borrowed stream are explicit. No host upload/download is performed.
    Standard/explicit CG, Jacobi, fusion, tolerance and error contracts follow
    solve_into; numerical failures return a status rather than a valid solution.
    Native computation releases the GIL and completes on stream before returning.

    Returns
    -------
    SolveResult
        GPU Vector plus status/counts. Download explicitly with result.x.to_host().
    """
    if not hasattr(b, "__cuda_array_interface__"):
        raise ValueError("b must be a device vector with __cuda_array_interface__")
    shape = tuple(b.__cuda_array_interface__["shape"])
    if len(shape) != 1:
        raise ValueError("b must be a one-dimensional device vector")
    x = Vector._empty(shape[0], device, stream)
    if x0 is None:
        x.fill(0.0, stream=stream)
    else:
        x.copy_from_device(x0, stream=stream)
    info = solve_into(
        A,
        b,
        x,
        workspace=Workspace(shape[0], device=device, stream=stream),
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
