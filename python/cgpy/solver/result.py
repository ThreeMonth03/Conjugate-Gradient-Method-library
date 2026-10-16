"""Convergence information shared by CPU and CUDA solvers."""

import dataclasses

from .. import _core


@dataclasses.dataclass(frozen=True, eq=False)
class SolveResult[Solution]:
    """Owned solution and independently checked native convergence data.

    ``x`` is a NumPy array for CPU solves or a CUDA Vector for device solves.
    Fields cannot be rebound; solution contents remain writable. Results
    compare by object identity; compare solution values explicitly. Check
    ``converged`` before using an approximation.
    """

    x: Solution
    info: _core.SolveInfo

    @property
    def converged(self) -> bool:
        """Whether the true residual satisfies the requested tolerance."""
        return self.info.converged

    @property
    def status(self) -> str:
        """Convergence, iteration-limit or numerical-failure status."""
        return self.info.status

    @property
    def iterations(self) -> int:
        """Number of completed solution updates."""
        return self.info.iterations

    @property
    def matvecs(self) -> int:
        """Matrix-vector products, including true-residual checks."""
        return self.info.matvecs

    @property
    def residual_norm(self) -> float:
        """Independently evaluated final Euclidean residual norm."""
        return self.info.residual_norm
