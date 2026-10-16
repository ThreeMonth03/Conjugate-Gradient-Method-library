"""Explicit CUDA execution-resource reuse without operator preparation."""

from threading import Lock

from ... import _core
from ...storage import cuda as storage
from .. import result
from . import api
from .workspace import Workspace


class Session:
    """Reuse one CUDA workspace while returning a fresh solution each call.

    Parameters
    ----------
    size : int
        Fixed system dimension.
    device : int, default=0
        Fixed CUDA device ordinal.
    stream : int, default=0
        Borrowed stream handle, which must outlive the session.
    dense_kernel : {'native', 'cublas'}, default='native'
        Fixed backend. CSR matrices require native.
    execution_mode : {'host', 'device'}, default='host'
        Fixed scalar-control engine, independent of fusion.

    Notes
    -----
    Owns an existing Workspace, including its vectors and backend resources.
    Matrix inputs are borrowed per call; no matrix or Jacobi inverse is cached.
    Each solve allocates an independent output and constructs Jacobi when
    selected. Use prepare to snapshot a matrix and cache Jacobi instead.
    Calls complete synchronously. Overlapping or reentrant calls on the same
    session raise ValueError; independent sessions own independent resources.
    Construction and destruction are outside session-scope benchmark timing.
    """

    def __init__(
        self,
        size: int,
        *,
        device: int = 0,
        stream: int = 0,
        dense_kernel: str = "native",
        execution_mode: str = "host",
    ) -> None:
        self._workspace = Workspace(
            size,
            device=device,
            stream=stream,
            dense_kernel=dense_kernel,
            execution_mode=execution_mode,
        )
        self._active = Lock()

    @property
    def size(self) -> int:
        """int: Fixed system dimension."""
        return self._workspace.size

    @property
    def device(self) -> int:
        """int: Fixed CUDA device ordinal."""
        return self._workspace.device

    @property
    def stream(self) -> int:
        """int: Borrowed stream used by all calls."""
        return self._workspace.stream

    @property
    def dense_kernel(self) -> str:
        """str: Fixed native or cuBLAS backend."""
        return self._workspace.dense_kernel

    @property
    def execution_mode(self) -> str:
        """str: Fixed host or device scalar-control engine."""
        return self._workspace.execution_mode

    def solve(
        self,
        A: object,
        b: object,
        x0: object | None = None,
        *,
        rtol: float = 1e-8,
        atol: float = 0.0,
        maxiter: int = 1000,
        algorithm: str = "standard",
        fused: bool = True,
        residual_refresh: int = 50,
        preconditioner: str = "none",
    ) -> result.SolveResult[storage.Vector]:
        """Solve a resident system using fresh output and per-call setup.

        Parameters
        ----------
        A : DenseMatrix, CsrMatrix or CUDA Array Interface object
            Borrowed nonempty square SPD matrix matching size/device. Different
            matrices may be supplied on successive calls; inputs must remain
            free of conflicting writes until the synchronous call returns.
        b, x0 : Vector or CUDA Array Interface object
            Contiguous float64 vectors. Omitted x0 selects zero. Its reset or
            device copy is included in each call; inputs are preserved.
        rtol, atol : float
            Finite nonnegative true-residual tolerances.
        maxiter : int
            Nonnegative update limit; zero checks only the initial guess.
        algorithm : {'standard', 'explicit'}, default='standard'
            Existing CG policy; Jacobi requires standard.
        fused : bool, default=True
            Select fused or separate native operations.
        residual_refresh : int, default=50
            Standard residual replacement interval; zero disables periodic
            replacement, retaining candidate/final true-residual checks.
        preconditioner : {'none', 'jacobi'}, default='none'
            Recomputed from this call's matrix, never retained by the session.

        Returns
        -------
        SolveResult
            Independent owning GPU Vector plus status and operation counts.
            Earlier results remain valid after another call or session deletion.

        Raises
        ------
        ValueError
            Invalid inputs/options, mismatched size/device, unsupported CSR
            backend, or overlapping/reentrant use of this session.
        RuntimeError
            CUDA allocation, stream or execution failure.

        Notes
        -----
        Device/stream/kernel/execution mode are fixed at construction. Version-2
        producers must be ready on the borrowed stream; version-3 dependencies
        are honored. No upload, download, CPU fallback or hidden pool occurs.
        Numerical failures use the same returned statuses as cuda.solve.
        """
        if not self._active.acquire(blocking=False):
            raise ValueError("CUDA session is already in use by another solve")
        try:
            maxiter, residual_refresh = _core._validate_iteration_limits(
                maxiter, residual_refresh
            )
            x, _ = api._initial_output(b, x0, self.device, self.stream)
            info = api.solve_into(
                A,
                b,
                x,
                workspace=self._workspace,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
                algorithm=algorithm,
                fused=fused,
                residual_refresh=residual_refresh,
                preconditioner=preconditioner,
            )
            return result.SolveResult(x, info)
        finally:
            self._active.release()
