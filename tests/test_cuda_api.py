"""Exercise explicit CUDA modes, device ownership and stream/workspace contracts."""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cgpy
import numpy as np
import pytest
from cgpy import cuda


def test_cuda_module_is_importable_and_reports_actual_capability():
    assert isinstance(cuda.enabled(), bool)
    assert isinstance(cuda.is_available(), bool)
    assert cuda.build_info()["enabled"] == cuda.enabled()
    if os.environ.get("CG_REQUIRE_CUDA") == "1":
        assert cuda.is_available(), cuda.build_info()
    if not cuda.enabled():
        with pytest.raises(RuntimeError, match="CUDA"):
            cuda.Vector(2)
    assert not cuda.is_available(-1)


@pytest.fixture
def gpu():
    if not cuda.is_available():
        pytest.skip("requires the optional CUDA extension and a compatible GPU")
    return cuda


@pytest.fixture
def system():
    matrix = np.diag(np.geomspace(1.0, 30.0, 33))
    matrix += np.diag(np.full(32, -0.125), 1) + np.diag(np.full(32, -0.125), -1)
    truth = np.linspace(-1.0, 1.0, 33)
    return matrix, matrix @ truth, truth


def cpu_csr(matrix):
    from scipy.sparse import csr_matrix

    value = csr_matrix(matrix)
    return cgpy.CSRMatrix(value.indptr, value.indices, value.data, value.shape)


@pytest.mark.parametrize(
    "kind,kernel", [("dense", "native"), ("dense", "cublas"), ("csr", "native")]
)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "algorithm,preconditioner", [("standard", "none"), ("standard", "jacobi"), ("explicit", "none")]
)
def test_device_modes_solve_to_true_residual_and_preserve_inputs(
    gpu, system, kind, kernel, fused, algorithm, preconditioner
):
    matrix, rhs, truth = system
    operator = (
        gpu.DenseMatrix.from_host(matrix)
        if kind == "dense"
        else gpu.CSRMatrix.from_host(cpu_csr(matrix))
    )
    b = gpu.Vector.from_host(rhs)
    initial = gpu.Vector.from_host(np.full(rhs.shape, 0.25))
    result = gpu.solve(
        operator,
        b,
        initial,
        dense_kernel=kernel,
        fused=fused,
        algorithm=algorithm,
        preconditioner=preconditioner,
        rtol=1e-10,
        maxiter=300,
    )
    assert result.converged
    answer = result.x.to_host()
    assert math.hypot(*(rhs - matrix @ answer)) <= 1e-10 * math.hypot(*rhs)
    np.testing.assert_allclose(answer, truth, rtol=1e-7, atol=1e-8)
    np.testing.assert_array_equal(b.to_host(), rhs)
    np.testing.assert_array_equal(initial.to_host(), np.full(rhs.shape, 0.25))
    assert operator.shape == matrix.shape
    assert result.x.device == operator.device == 0


@pytest.mark.parametrize("kernel", ["native", "cublas"])
def test_prepared_snapshot_and_workspace_reuse(gpu, system, kernel):
    matrix, rhs, truth = system
    plan = gpu.prepare_host(matrix, dense_kernel=kernel, preconditioner="jacobi")
    matrix.fill(np.nan)
    b = gpu.Vector.from_host(rhs)
    x = plan.create_vector()
    workspace = plan.create_workspace()
    assert plan.dense_kernel == kernel
    assert plan.preconditioner == "jacobi"
    assert plan.preconditioner_nbytes == rhs.nbytes
    assert plan.preconditioner_setup_ns > 0
    for _ in range(2):
        x.fill(0.0)
        info = plan.solve_into(b, x, workspace=workspace, rtol=1e-10)
        assert info.converged and info.iterations > 0
        np.testing.assert_allclose(x.to_host(), truth, rtol=1e-7, atol=1e-8)
        reused = plan.solve_into(b, x, workspace=workspace, rtol=1e-10)
        assert reused.converged and reused.iterations == 0 and reused.matvecs == 1


def test_unsupported_modes_layouts_aliases_and_workspaces_reject_before_writes(gpu, system):
    matrix, rhs, _ = system
    A = gpu.DenseMatrix.from_host(matrix)
    b = gpu.Vector.from_host(rhs)
    x = gpu.Vector(rhs.size)
    workspace = gpu.Workspace(rhs.size)
    for mode in ("auto", "blas", "unknown"):
        with pytest.raises(ValueError, match="dense_kernel"):
            gpu.prepare(A, dense_kernel=mode)
        with pytest.raises(ValueError, match="dense_kernel"):
            gpu.solve_into(A, b, x, workspace=workspace, dense_kernel=mode)
    with pytest.raises(ValueError, match="CSR|native"):
        gpu.prepare(gpu.CSRMatrix.from_host(cpu_csr(matrix)), dense_kernel="cublas")
    with pytest.raises(ValueError):
        gpu.solve_into(A, b, x, workspace=gpu.Workspace(rhs.size - 1))
    with pytest.raises(ValueError):
        gpu.solve_into(A, b, b, workspace=workspace)
    with pytest.raises(ValueError, match="device"):
        gpu.solve_into(A, rhs, x, workspace=workspace)
    with pytest.raises(ValueError):
        gpu.solve_into(A, b, x, workspace=workspace, maxiter=-1)
    with pytest.raises(ValueError):
        gpu.solve_into(A, b, x, workspace=workspace, rtol=float("nan"))
    with pytest.raises(ValueError):
        gpu.solve_into(A, b, x, workspace=workspace, preconditioner="jacobi", algorithm="explicit")
    np.testing.assert_array_equal(x.to_host(), np.zeros_like(rhs))


@pytest.mark.parametrize("kernel", ["native", "cublas"])
def test_true_norm_and_numerical_failure_statuses(gpu, kernel):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    for magnitude in (1e-200, 1e200):
        b = gpu.Vector.from_host(np.array([magnitude, -magnitude]))
        limited = gpu.solve(A, b, dense_kernel=kernel, maxiter=0)
        assert limited.status == "max_iterations"
        assert math.isclose(limited.residual_norm, math.hypot(magnitude, magnitude), rel_tol=1e-14)
        failed = gpu.solve(A, b, dense_kernel=kernel)
        assert failed.status == "non_finite"
        assert math.isclose(failed.residual_norm, math.hypot(magnitude, magnitude), rel_tol=1e-14)
    for magnitude in (1e-150, 1e150):
        b = gpu.Vector.from_host(np.array([magnitude, -magnitude]))
        result = gpu.solve(A, b, dense_kernel=kernel)
        assert result.converged
        np.testing.assert_array_equal(result.x.to_host(), b.to_host())
    for bad in (np.nan, np.inf):
        invalid = gpu.DenseMatrix.from_host(np.array([[bad, 0.0], [0.0, 1.0]]))
        b = gpu.Vector.from_host([1.0, 1.0])
        assert gpu.solve(invalid, b, dense_kernel=kernel).status == "non_finite"


def torch_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA PyTorch")
    return torch


def test_cuda_interface_rejects_host_pointers_and_ambiguous_producer_stream(gpu, system):
    matrix, rhs, _ = system
    plan = gpu.prepare_host(matrix)
    output = plan.create_vector()
    workspace = plan.create_workspace()

    class Export:
        def __init__(self, descriptor):
            self.__cuda_array_interface__ = descriptor

    host_pointer = Export(
        {
            "version": 3,
            "typestr": "<f8",
            "shape": rhs.shape,
            "strides": None,
            "data": (rhs.ctypes.data, False),
            "stream": None,
        }
    )
    with pytest.raises(ValueError, match="device storage|requested device"):
        plan.solve_into(host_pointer, output, workspace=workspace)
    device_rhs = gpu.Vector.from_host(rhs)
    ambiguous = Export({**device_rhs.__cuda_array_interface__, "stream": 0})
    with pytest.raises(ValueError, match="stream"):
        plan.solve_into(ambiguous, output, workspace=workspace)
    # Validation failures leave the CUDA runtime and output usable.
    assert plan.solve_into(device_rhs, output, workspace=workspace).converged


def test_torch_zero_copy_strict_layout_and_borrowed_nondefault_stream(gpu, system):
    torch = torch_gpu()
    matrix, rhs, truth = system
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        A = torch.tensor(matrix, dtype=torch.float64, device="cuda")
        b = torch.tensor(rhs, dtype=torch.float64, device="cuda")
        output = torch.zeros_like(b)
        plan = gpu.prepare(A, stream=stream.cuda_stream)
        workspace = plan.create_workspace(stream=stream.cuda_stream)
        current = torch.cuda.current_stream().cuda_stream
        info = plan.solve_into(b, output, workspace=workspace, rtol=1e-10)
        assert info.converged
        assert torch.cuda.current_stream().cuda_stream == current
        np.testing.assert_allclose(output.cpu().numpy(), truth, rtol=1e-7, atol=1e-8)
        for bad in (b.float(), torch.empty(b.size(0) * 2, device="cuda", dtype=torch.float64)[::2]):
            with pytest.raises(ValueError):
                plan.solve_into(bad, output, workspace=workspace)
        with pytest.raises(ValueError):
            gpu.solve_into(A, b, A[0], workspace=workspace)
        A.fill_(np.nan)
        output.zero_()
        assert plan.solve_into(b, output, workspace=workspace).converged


def test_version3_producer_dependencies_and_readonly_outputs(gpu, system):
    torch = torch_gpu()
    matrix, rhs, truth = system
    producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()

    class Export:
        def __init__(self, tensor, readonly=False):
            self.owner = tensor
            self.__cuda_array_interface__ = {
                **tensor.__cuda_array_interface__,
                "version": 3,
                "stream": producer.cuda_stream,
                "data": (tensor.data_ptr(), readonly),
            }

    with torch.cuda.stream(producer):
        A = torch.tensor(matrix, device="cuda", dtype=torch.float64) * 1.0
        b = torch.tensor(rhs, device="cuda", dtype=torch.float64) * 1.0
    plan = gpu.prepare(Export(A), stream=consumer.cuda_stream)
    x = plan.create_vector(stream=consumer.cuda_stream)
    workspace = plan.create_workspace(stream=consumer.cuda_stream)
    assert plan.solve_into(Export(b, True), x, workspace=workspace, rtol=1e-10).converged
    np.testing.assert_allclose(x.to_host(), truth, rtol=1e-7, atol=1e-8)
    with pytest.raises(ValueError, match="writable"):
        plan.solve_into(Export(b), Export(b, True), workspace=workspace)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_device_csr_snapshot_validates_canonical_indices(gpu, dtype):
    torch = torch_gpu()
    index_dtype = getattr(torch, dtype)
    pointers = torch.tensor([0, 2, 4], device="cuda", dtype=index_dtype)
    columns = torch.tensor([0, 1, 0, 1], device="cuda", dtype=index_dtype)
    values = torch.tensor([4.0, 1.0, 1.0, 3.0], device="cuda", dtype=torch.float64)
    matrix = gpu.CSRMatrix.from_device(pointers, columns, values, (2, 2))
    assert matrix.nnz == 4
    values.fill_(np.nan)
    answer = gpu.solve(matrix, gpu.Vector.from_host([1.0, 2.0]), preconditioner="jacobi")
    assert answer.converged
    np.testing.assert_allclose(answer.x.to_host(), [1 / 11, 7 / 11])
    for bad in ([1, 0, 0, 1], [0, 0, 0, 1], [0, 2, 0, 1]):
        with pytest.raises(ValueError):
            gpu.CSRMatrix.from_device(
                pointers,
                torch.tensor(bad, device="cuda", dtype=index_dtype),
                torch.ones(4, device="cuda", dtype=torch.float64),
                (2, 2),
            )


@pytest.mark.parametrize("kernel", ["native", "cublas"])
def test_shared_immutable_plan_with_independent_concurrent_workspaces(gpu, system, kernel):
    torch = torch_gpu()
    matrix, rhs, truth = system
    plan = gpu.prepare_host(matrix, dense_kernel=kernel, preconditioner="jacobi")
    b = gpu.Vector.from_host(rhs)
    streams = [torch.cuda.Stream() for _ in range(2)]

    def run(index):
        stream = streams[index]
        workspace = plan.create_workspace(stream=stream.cuda_stream)
        x = plan.create_vector(stream=stream.cuda_stream)
        assert plan.solve_into(b, x, workspace=workspace, rtol=1e-10).converged
        np.testing.assert_allclose(
            x.to_host(stream=stream.cuda_stream), truth, rtol=1e-7, atol=1e-8
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))
