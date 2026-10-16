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
    cuda.PreparedSolver()  # type: ignore[call-arg]
    cuda.DenseMatrix()  # type: ignore[call-arg]
    cuda.CsrMatrix()  # type: ignore[call-arg]
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


def invalid_cuda_exporters(
    matrix: cuda.DenseMatrix,
    sparse: cuda.CsrMatrix,
    rhs: cuda.Vector,
    output: cuda.Vector,
    host_array: npt.NDArray[np.float64],
    plain_object: object,
    workspace: cuda.Workspace,
) -> None:
    cuda.prepare(host_array)  # type: ignore[arg-type]
    cuda.prepare([1.0, 2.0])  # type: ignore[arg-type]
    cuda.prepare(1)  # type: ignore[arg-type]
    cuda.prepare(plain_object)  # type: ignore[arg-type]
    cuda.solve(matrix, [1.0, 2.0])  # type: ignore[arg-type]
    cuda.solve(matrix, host_array)  # type: ignore[arg-type]
    cuda.solve(matrix, rhs, plain_object)  # type: ignore[arg-type]
    cuda.solve_into(matrix, rhs, [0.0, 0.0], workspace=workspace)  # type: ignore[arg-type]
    cuda.prepare(matrix).solve_into(matrix, output, workspace=workspace)  # type: ignore[arg-type]
    cuda.prepare(matrix).solve_into(rhs, sparse, workspace=workspace)  # type: ignore[arg-type]
    cuda.Session(2).solve(matrix, host_array)  # type: ignore[arg-type]
    rhs.copy_from_device(matrix)  # type: ignore[arg-type]
    rhs.copy_from_device(host_array)  # type: ignore[arg-type]
    cuda.DenseMatrix.from_device(matrix)  # type: ignore[arg-type]
    cuda.DenseMatrix.from_device(plain_object)  # type: ignore[arg-type]
    cuda.CsrMatrix.from_device(sparse, rhs, output, (2, 2))  # type: ignore[arg-type]
    cuda.CsrMatrix.from_device(rhs, host_array, output, (2, 2))  # type: ignore[arg-type]
    cuda.CsrMatrix.from_device(rhs, rhs, [1.0, 2.0], (2, 2))  # type: ignore[arg-type]
