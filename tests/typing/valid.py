"""Public API examples checked statically, never executed."""

from typing import Literal, TypedDict, assert_type

import cgpy
import numpy as np
import numpy.typing as npt
from cgpy import Workspace as CpuWorkspace
from cgpy import cuda


class DeviceArray:
    @property
    def __cuda_array_interface__(self) -> object:
        return {}


class AttributeDeviceArray:
    __cuda_array_interface__: dict[str, object]


class DeviceDescriptor(TypedDict):
    shape: tuple[int, ...]
    typestr: str
    data: tuple[int, bool]
    version: int


class TypedDeviceArray:
    @property
    def __cuda_array_interface__(self) -> DeviceDescriptor:
        return {
            "shape": (2,),
            "typestr": "<f8",
            "data": (1, False),
            "version": 3,
        }


def cpu_examples(
    matrix: npt.NDArray[np.float64],
    rhs: npt.NDArray[np.float64],
    output: npt.NDArray[np.float64],
) -> None:
    sparse = cgpy.CsrMatrix([0, 1, 2], [0, 1], [2.0, 3.0], (2, 2))
    assert_type(sparse.shape, tuple[int, int])
    assert_type(sparse.nnz, int)
    plan = assert_type(cgpy.prepare(matrix), cgpy.PreparedSolver)
    assert_type(cgpy.prepare(sparse), cgpy.PreparedSolver)
    assert_type(cgpy.prepare([[2.0, 0.0], [0.0, 3.0]]), cgpy.PreparedSolver)
    workspace = assert_type(plan.create_workspace(), CpuWorkspace)
    assert_type(CpuWorkspace(2), CpuWorkspace)
    assert_type(workspace.size, int)
    assert_type(workspace.capacity, int)
    assert_type(plan.size, int)
    assert_type(plan.num_threads, int)
    assert_type(plan.dense_kernel, Literal["native", "blas"])
    assert_type(plan.preconditioner, Literal["none", "jacobi"])
    info = assert_type(
        plan.solve_into(
            rhs,
            output,
            workspace=workspace,
            maxiter=np.int64(1000),
            residual_refresh=np.int64(50),
        ),
        cgpy.SolveInfo,
    )
    assert_type(info.converged, bool)
    assert_type(info.iterations, int)
    assert_type(info.matvecs, int)
    assert_type(info.restarts, int)
    assert_type(info.residual_norm, float)
    assert_type(
        info.status,
        Literal[
            "converged",
            "max_iterations",
            "non_positive_curvature",
            "non_finite",
        ],
    )
    assert_type(
        cgpy.solve_into(matrix, rhs, output, workspace=workspace),
        cgpy.SolveInfo,
    )
    result = assert_type(
        cgpy.solve(matrix, rhs, algorithm="explicit", residual_refresh=0),
        cgpy.SolveResult[npt.NDArray[np.float64]],
    )
    assert_type(result.x, npt.NDArray[np.float64])
    assert_type(result.info, cgpy.SolveInfo)
    assert_type(cgpy.solve(sparse, [1.0, 2.0]).x, npt.NDArray[np.float64])


def cuda_examples(
    host_matrix: npt.NDArray[np.float64],
    host_rhs: npt.NDArray[np.float64],
    cpu_sparse: cgpy.CsrMatrix,
) -> None:
    matrix = assert_type(
        cuda.DenseMatrix.from_host(host_matrix), cuda.DenseMatrix
    )
    sparse = assert_type(cuda.CsrMatrix.from_host(cpu_sparse), cuda.CsrMatrix)
    rhs = assert_type(cuda.Vector.from_host(host_rhs), cuda.Vector)
    assert_type(rhs.shape, tuple[int])
    assert_type(matrix.shape, tuple[int, int])
    assert_type(sparse.shape, tuple[int, int])
    assert_type(rhs.to_host(), npt.NDArray[np.float64])
    plan = assert_type(cuda.prepare(matrix), cuda.PreparedSolver)
    assert_type(cuda.prepare(sparse), cuda.PreparedSolver)
    assert_type(cuda.prepare_host(host_matrix), cuda.PreparedSolver)
    assert_type(cuda.prepare_host(cpu_sparse), cuda.PreparedSolver)
    workspace = assert_type(
        plan.create_workspace(execution_mode="device"), cuda.Workspace
    )
    assert_type(workspace.execution_mode, Literal["host", "device"])
    assert_type(workspace.dense_kernel, Literal["native", "cublas"])
    output = assert_type(plan.create_vector(), cuda.Vector)
    assert_type(
        plan.solve_into(rhs, output, workspace=workspace), cgpy.SolveInfo
    )
    assert_type(
        cuda.solve_into(matrix, rhs, output, workspace=workspace),
        cgpy.SolveInfo,
    )
    result = assert_type(
        cuda.solve(matrix, rhs), cgpy.SolveResult[cuda.Vector]
    )
    assert_type(result.x, cuda.Vector)
    assert_type(result.x.to_host(), npt.NDArray[np.float64])
    session = cuda.Session(2, dense_kernel="cublas", execution_mode="device")
    assert_type(session.solve(matrix, rhs), cgpy.SolveResult[cuda.Vector])
    assert_type(cuda.solve(sparse, rhs), cgpy.SolveResult[cuda.Vector])
    assert_type(
        cuda.solve_into(sparse, rhs, output, workspace=workspace),
        cgpy.SolveInfo,
    )
    assert_type(session.solve(sparse, rhs), cgpy.SolveResult[cuda.Vector])


def cuda_exporter_examples(
    matrix: DeviceArray,
    rhs: AttributeDeviceArray,
    output: TypedDeviceArray,
    vector: cuda.Vector,
    plan: cuda.PreparedSolver,
    workspace: cuda.Workspace,
    session: cuda.Session,
) -> None:
    exporters: tuple[cuda.SupportsCudaArrayInterface, ...] = (
        matrix,
        rhs,
        output,
        vector,
    )
    for exporter in exporters:
        assert_type(exporter.__cuda_array_interface__, object)
    assert_type(cuda.DenseMatrix.from_device(matrix), cuda.DenseMatrix)
    assert_type(
        cuda.CsrMatrix.from_device(matrix, rhs, output, (2, 2)), cuda.CsrMatrix
    )
    vector.copy_from_device(rhs)
    vector.copy_from_device(output)
    assert_type(cuda.prepare(matrix), cuda.PreparedSolver)
    assert_type(
        cuda.solve_into(matrix, rhs, output, workspace=workspace),
        cgpy.SolveInfo,
    )
    assert_type(cuda.solve(matrix, rhs, output), cgpy.SolveResult[cuda.Vector])
    assert_type(
        plan.solve_into(rhs, output, workspace=workspace), cgpy.SolveInfo
    )
    assert_type(
        session.solve(matrix, rhs, output), cgpy.SolveResult[cuda.Vector]
    )
