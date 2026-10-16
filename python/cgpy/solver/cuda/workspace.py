"""Reusable CUDA solver vectors, reductions and execution resources."""

from ..._runtime import cuda as cuda_runtime


class Workspace:
    """Own solver vectors, reduction scratch and selected backend resources.

    Parameters
    ----------
    size : int
        Matrix dimension. Storage is allocated once and reused across solves.
    device : int, default=0
        Device ordinal, fixed at construction.
    stream : int, default=0
        Borrowed stream handle, fixed at construction; it must outlive this
        workspace. One workspace/output pair belongs to one active solve.
    dense_kernel : {'native', 'cublas'}, default='native'
        Fixed kernel selection. Only cublas allocates a handle and its
        additional workspace; all resources are prepared at construction.
    execution_mode : {'host', 'device'}, default='host'
        Fixed native scalar-control engine. Host reads individual reductions;
        device keeps recurrence scalars on the GPU and reads completed status
        packets. Both modes finish on the workspace stream before returning.

    Notes
    -----
    Device, stream, kernel and execution mode come from this workspace when
    calling solve_into. A prepared solver additionally requires a matching
    kernel because its matrix setup fixes that choice.
    Distinct workspaces permit overlapping calls against an immutable plan.
    Each workspace owns its backend resources and reduction storage. Reuse on the same
    stream is safe because public solves return only after completion.
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
        self._native = cuda_runtime.load().Workspace(
            size, device, stream, dense_kernel, execution_mode
        )

    @property
    def size(self) -> int:
        """int: Fixed system dimension."""
        return self._native.size

    @property
    def device(self) -> int:
        """int: Fixed device ordinal."""
        return self._native.device

    @property
    def stream(self) -> int:
        """int: Borrowed stream handle used by all workspace operations."""
        return self._native.stream

    @property
    def dense_kernel(self) -> str:
        """str: Kernel whose resources were prepared at construction."""
        return self._native.dense_kernel

    @property
    def execution_mode(self) -> str:
        """str: Fixed host or device scalar-control engine."""
        return self._native.execution_mode
