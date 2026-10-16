"""Immutable prepared CUDA solver façade and matching resource factories."""

from ... import _core
from ..._interop import cuda as cuda_interop
from ...storage import cuda as storage
from . import workspace as workspace_module


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

    def create_workspace(
        self, *, stream: int = 0, execution_mode: str = "host"
    ) -> workspace_module.Workspace:
        """Allocate storage with an explicit stream and scalar-control engine.

        execution_mode selects host (default) or device and is fixed for this
        workspace. The immutable plan can serve either mode with distinct
        workspaces; solves have no redundant execution-mode argument.
        """
        return workspace_module.Workspace(
            self.size,
            device=self.device,
            stream=stream,
            dense_kernel=self.dense_kernel,
            execution_mode=execution_mode,
        )

    def create_vector(self, *, stream: int = 0) -> storage.Vector:
        """Allocate an independent zero output on this plan's device."""
        return storage.Vector(self.size, device=self.device, stream=stream)

    def solve_into(
        self,
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
    ) -> _core.SolveInfo:
        """Solve with cached setup and caller-owned GPU output/workspace.

        Parameters
        ----------
        b, x : Vector or CUDA Array Interface object
            Contiguous float64 device vectors. x is writable initial guess and
            output, disjoint from b. Reset x before a repeated comparison.
        workspace : Workspace
            Matching size/device/kernel, unused by overlapping calls. Its stream
            determines execution and the version-2 producer readiness contract.
            Its execution_mode selects the native host/device scalar engine.
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
        return self._native.solve_into(
            cuda_interop.native_object(b),
            cuda_interop.native_object(x),
            workspace=workspace._native,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
            algorithm=algorithm,
            fused=fused,
            residual_refresh=residual_refresh,
        )
