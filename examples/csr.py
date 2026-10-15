"""Solve a canonical CSR system without constructing a dense matrix."""

import cgpy
import numpy as np


def main() -> None:
    matrix = cgpy.CsrMatrix(
        indptr=[0, 2, 5, 7],
        indices=[0, 1, 0, 1, 2, 1, 2],
        data=[2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0],
        shape=(3, 3),
    )
    result = cgpy.solve(matrix, [1.0, 0.0, 1.0], rtol=1e-12)
    if not result.converged:
        raise RuntimeError(f"CG failed: {result.status}")
    np.testing.assert_allclose(result.x, np.ones(3), rtol=1e-12)
    print(f"CSR CG: x={result.x}, iterations={result.iterations}")


if __name__ == "__main__":
    main()
