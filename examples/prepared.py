"""Reuse matrix setup, Jacobi setup and workspace across right-hand sides."""

import cgpy
import numpy as np


def main() -> None:
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    plan = cgpy.prepare(matrix, preconditioner="jacobi")
    workspace = plan.create_workspace()
    solution = np.zeros(2)
    for rhs in (np.array([1.0, 2.0]), np.array([3.0, 4.0])):
        solution.fill(0.0)
        info = plan.solve_into(rhs, solution, workspace=workspace, rtol=1e-12)
        if not info.converged:
            raise RuntimeError(f"PCG failed: {info.status}")
        np.testing.assert_allclose(matrix @ solution, rhs, rtol=1e-12)
        print(f"Prepared PCG: b={rhs}, x={solution}")


if __name__ == "__main__":
    main()
