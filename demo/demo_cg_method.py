"""Solve a small SPD system with the installed public CPU API."""

import cgpy
import numpy as np


def main():
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    result = cgpy.solve(matrix, rhs, rtol=1e-10)
    assert result.converged, result.status
    np.testing.assert_allclose(result.x, np.linalg.solve(matrix, rhs))
    print(f"status={result.status} iterations={result.iterations}")
    print(f"x={result.x} true_residual={result.residual_norm:.6g}")


if __name__ == "__main__":
    main()
