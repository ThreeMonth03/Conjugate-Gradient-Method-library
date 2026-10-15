"""Reusable CUDA solver vectors, reductions and execution resources."""

from .._runtime import cuda as cuda_runtime


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

    def __init__(self, size: int, *, device: int = 0, stream: int = 0) -> None:
        self._native = cuda_runtime.load().Workspace(size, device, stream)

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
