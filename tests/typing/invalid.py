"""Each precise ignore must suppress a real public contract violation."""

import cgpy
import numpy as np
import numpy.typing as npt
from cgpy import cuda


def invalid_cpu_calls(
    matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    output: npt.NDArray[np.float64],
    cuda_workspace: cuda.Workspace,
) -> None:
    cgpy.PreparedSolver()  # type: ignore[call-arg]
    cgpy.SolveInfo()  # type: ignore[call-arg]
    cgpy.solve(matrix, rhs, algorithm="standrad")  # type: ignore[arg-type]
    cgpy.prepare(matrix, preconditioner="Jacobi")  # type: ignore[arg-type]
    cgpy.prepare(matrix, dense_kernel="cublas")  # type: ignore[arg-type]
    cgpy.solve_into(
        matrix,
        rhs,
        output,
        workspace=cuda_workspace,  # type: ignore[arg-type]
    )
    cgpy.solve_into(matrix, rhs, [0.0, 0.0], workspace=cgpy.Workspace(2))  # type: ignore[arg-type]
    cgpy.solve(matrix, rhs, maxiter=1.5)  # type: ignore[arg-type]
    cgpy.prepare(matrix).size = 2  # type: ignore[misc]
    cgpy.Workspace(2).size = 2  # type: ignore[misc]
    result = cgpy.solve(matrix, rhs)
    result.info.iterations = 0  # type: ignore[misc]
    result.info.status = "converged"  # type: ignore[misc]
    result.x = output  # type: ignore[misc]


def invalid_cuda_calls(
    matrix: cuda.DenseMatrix,
    rhs: cuda.Vector,
    output: cuda.Vector,
    cpu_workspace: cgpy.Workspace,
) -> None:
    cuda.solve(matrix, rhs, algorithm="standrad")  # type: ignore[arg-type]
    cuda.prepare(matrix, dense_kernel="blas")  # type: ignore[arg-type]
    cuda.Workspace(2, execution_mode="automatic")  # type: ignore[arg-type]
    cuda.solve_into(
        matrix,
        rhs,
        output,
        workspace=cpu_workspace,  # type: ignore[arg-type]
    )
    cuda.prepare(matrix).size = 2  # type: ignore[misc]
    cuda.Workspace(2).execution_mode = "host"  # type: ignore[misc]
    cuda.CsrMatrix.from_host([[1.0]])  # type: ignore[arg-type]
    rhs.fill("zero")  # type: ignore[arg-type]
