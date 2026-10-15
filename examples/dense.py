"""Solve a dense SPD system with Jacobi preconditioned CG."""

import cgpy
import numpy as np


def main() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    result = cgpy.solve(matrix, rhs, preconditioner="jacobi", rtol=1e-12)
    if not result.converged:
        raise RuntimeError(f"CG failed: {result.status}")
    np.testing.assert_allclose(matrix @ result.x, rhs, rtol=1e-12)
    print(f"Dense PCG: x={result.x}, residual={result.residual_norm:.3e}")


if __name__ == "__main__":
    main()
