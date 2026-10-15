"""Exercise explicit CUDA modes, device ownership and stream/workspace contracts."""

import math
import os
from concurrent.futures import ThreadPoolExecutor

import cgpy
import numpy as np
import pytest
from cgpy import cuda


def test_cuda_module_is_importable_and_reports_actual_capability():
    assert cuda.CsrMatrix.__name__ == "CsrMatrix"
    assert not hasattr(cuda, "CSRMatrix")
    assert not hasattr(cuda, "PreparedCsrSolver")
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
        pytest.skip(
            "requires the optional CUDA extension and a compatible GPU"
        )
    return cuda


@pytest.fixture
def system():
    matrix = np.diag(np.geomspace(1.0, 30.0, 33))
    matrix += np.diag(np.full(32, -0.125), 1) + np.diag(
        np.full(32, -0.125), -1
    )
    truth = np.linspace(-1.0, 1.0, 33)
    return matrix, matrix @ truth, truth


class IndexCount:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("entry", ["convenience", "strict", "prepared"])
@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
def test_cuda_integer_protocol_is_evaluated_once_per_option(
    gpu, entry, option
):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    count = IndexCount(2)
    options = {option: count}
    if entry == "convenience":
        assert gpu.solve(A, b, **options).converged
    else:
        plan = gpu.prepare(A)
        workspace = plan.create_workspace()
        x = plan.create_vector()
        info = (
            gpu.solve_into(A, b, x, workspace=workspace, **options)
            if entry == "strict"
            else plan.solve_into(b, x, workspace=workspace, **options)
        )
        assert info.converged
    assert count.calls == 1


@pytest.mark.parametrize("entry", ["strict", "prepared"])
@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
def test_cuda_validates_output_after_integer_protocol_callbacks(
    gpu, entry, option
):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    plan = gpu.prepare(A)
    workspace = plan.create_workspace()
    storage = plan.create_vector()

    class OutputView:
        owner = storage
        readonly = False

        @property
        def __cuda_array_interface__(self):
            interface = dict(self.owner.__cuda_array_interface__)
            interface["data"] = (interface["data"][0], self.readonly)
            return interface

    output = OutputView()

    class FreezeOutput(IndexCount):
        def __index__(self):
            output.readonly = True
            return super().__index__()

    count = FreezeOutput(2)
    with pytest.raises(ValueError, match="writable"):
        if entry == "strict":
            gpu.solve_into(
                A, b, output, workspace=workspace, **{option: count}
            )
        else:
            plan.solve_into(b, output, workspace=workspace, **{option: count})
    assert count.calls == 1
    np.testing.assert_array_equal(storage.to_host(), np.zeros(2))
    output.readonly = False
    assert plan.solve_into(b, output, workspace=workspace).converged
    np.testing.assert_array_equal(storage.to_host(), np.ones(2))


@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
@pytest.mark.parametrize("value", [-1, 1.5, 1 << 100, None, "1"])
def test_cuda_iteration_errors_preserve_output_and_workspace(
    gpu, option, value
):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    plan = gpu.prepare(A)
    workspace = plan.create_workspace()
    x = plan.create_vector()
    with pytest.raises(ValueError, match=option):
        gpu.solve(A, b, **{option: value})
    with pytest.raises(ValueError, match=option):
        gpu.solve_into(A, b, x, workspace=workspace, **{option: value})
    with pytest.raises(ValueError, match=option):
        plan.solve_into(b, x, workspace=workspace, **{option: value})
    np.testing.assert_array_equal(x.to_host(), np.zeros(2))
    assert plan.solve_into(b, x, workspace=workspace).converged


@pytest.mark.parametrize(
    "count",
    [
        np.int64(2),
        np.uint64(2),
        IndexCount(2),
        True,
        int(np.iinfo(np.uintp).max),
    ],
)
def test_cuda_entry_points_accept_integer_index_protocol_and_native_max(
    gpu, count
):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    plan = gpu.prepare(A)
    workspace = plan.create_workspace()
    x = plan.create_vector()
    assert gpu.solve(A, b, maxiter=count, residual_refresh=count).converged
    assert gpu.solve_into(
        A, b, x, workspace=workspace, maxiter=count, residual_refresh=count
    ).converged
    x.fill(0)
    assert plan.solve_into(
        b, x, workspace=workspace, maxiter=count, residual_refresh=count
    ).converged
    np.testing.assert_allclose(x.to_host(), np.ones(2))


@pytest.mark.parametrize("count", [False, np.uint64(0), IndexCount(0)])
def test_cuda_zero_update_limit_preserves_initial_guess(gpu, count):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    plan = gpu.prepare(A)
    workspace = plan.create_workspace()
    x = plan.create_vector()
    results = [
        gpu.solve(A, b, maxiter=count).info,
        gpu.solve_into(A, b, x, workspace=workspace, maxiter=count),
        plan.solve_into(b, x, workspace=workspace, maxiter=count),
    ]
    assert all(info.status == "max_iterations" for info in results)
    assert all(info.iterations == 0 for info in results)
    np.testing.assert_array_equal(x.to_host(), np.zeros(2))


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_cuda_iteration_protocol_preserves_unexpected_exceptions(gpu, error):
    class BrokenIndex:
        def __index__(self):
            raise error("index sentinel")

    A = gpu.DenseMatrix.from_host(np.eye(2))
    b = gpu.Vector.from_host(np.ones(2))
    plan = gpu.prepare(A)
    workspace = plan.create_workspace()
    x = plan.create_vector()
    for call in (
        lambda: gpu.solve(A, b, maxiter=BrokenIndex()),
        lambda: gpu.solve_into(
            A, b, x, workspace=workspace, maxiter=BrokenIndex()
        ),
        lambda: plan.solve_into(
            b, x, workspace=workspace, maxiter=BrokenIndex()
        ),
    ):
        with pytest.raises(error, match="index sentinel"):
            call()
    np.testing.assert_array_equal(x.to_host(), np.zeros(2))
    assert plan.solve_into(b, x, workspace=workspace).converged


def test_cuda_host_upload_normalizes_foreign_storage_and_owns_snapshot(gpu):
    backing = np.ndarray(
        (6,), dtype=">f4", buffer=bytearray(6 * 4 + 1), offset=1
    )
    backing[:] = [1.0, 99.0, 2.0, 99.0, 3.0, 99.0]
    source = backing[::2]
    source.setflags(write=False)
    vector = gpu.Vector.from_host(source)
    backing.fill(0)
    np.testing.assert_array_equal(vector.to_host(), [1.0, 2.0, 3.0])
    vector.copy_from_host(np.array([3.0, 2.0, 1.0], dtype=np.int16))
    np.testing.assert_array_equal(vector.to_host(), [3.0, 2.0, 1.0])


def cpu_csr(matrix):
    from scipy.sparse import csr_matrix

    value = csr_matrix(matrix)
    return cgpy.CsrMatrix(value.indptr, value.indices, value.data, value.shape)


@pytest.mark.parametrize(
    "kind,kernel",
    [("dense", "native"), ("dense", "cublas"), ("csr", "native")],
)
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "algorithm,preconditioner",
    [("standard", "none"), ("standard", "jacobi"), ("explicit", "none")],
)
@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_device_modes_solve_to_true_residual_and_preserve_inputs(
    gpu, system, kind, kernel, fused, algorithm, preconditioner, execution_mode
):
    matrix, rhs, truth = system
    operator = (
        gpu.DenseMatrix.from_host(matrix)
        if kind == "dense"
        else gpu.CsrMatrix.from_host(cpu_csr(matrix))
    )
    b = gpu.Vector.from_host(rhs)
    initial = gpu.Vector.from_host(np.full(rhs.shape, 0.25))
    result = gpu.solve(
        operator,
        b,
        initial,
        dense_kernel=kernel,
        execution_mode=execution_mode,
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
@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_prepared_snapshot_and_workspace_reuse(
    gpu, system, kernel, execution_mode
):
    matrix, rhs, truth = system
    plan = gpu.prepare_host(
        matrix, dense_kernel=kernel, preconditioner="jacobi"
    )
    matrix.fill(np.nan)
    b = gpu.Vector.from_host(rhs)
    x = plan.create_vector()
    workspace = plan.create_workspace(execution_mode=execution_mode)
    assert workspace.execution_mode == execution_mode
    with pytest.raises(AttributeError):
        workspace.execution_mode = (
            "device" if execution_mode == "host" else "host"
        )
    assert plan.dense_kernel == kernel
    assert workspace.dense_kernel == kernel
    assert plan.preconditioner == "jacobi"
    assert plan.preconditioner_nbytes == rhs.nbytes
    assert plan.preconditioner_setup_ns > 0
    for _ in range(2):
        x.fill(0.0)
        info = plan.solve_into(b, x, workspace=workspace, rtol=1e-10)
        assert info.converged and info.iterations > 0
        np.testing.assert_allclose(x.to_host(), truth, rtol=1e-7, atol=1e-8)
        reused = plan.solve_into(b, x, workspace=workspace, rtol=1e-10)
        assert (
            reused.converged and reused.iterations == 0 and reused.matvecs == 1
        )


@pytest.mark.parametrize("kernel", ["native", "cublas"])
def test_workspace_backend_mismatch_rejects_before_writes(gpu, system, kernel):
    matrix, rhs, truth = system
    plan = gpu.prepare_host(matrix, dense_kernel=kernel)
    operator = gpu.DenseMatrix.from_host(matrix)
    other = "cublas" if kernel == "native" else "native"
    mismatched = gpu.Workspace(rhs.size, dense_kernel=other)
    b, x = gpu.Vector.from_host(rhs), plan.create_vector()
    with pytest.raises(ValueError, match="kernel"):
        plan.solve_into(b, x, workspace=mismatched)
    np.testing.assert_array_equal(x.to_host(), np.zeros_like(rhs))
    with pytest.raises(ValueError, match="kernel"):
        gpu.solve_into(
            operator, b, x, workspace=mismatched, dense_kernel=kernel
        )
    np.testing.assert_array_equal(x.to_host(), np.zeros_like(rhs))
    workspace = plan.create_workspace()
    assert plan.solve_into(b, x, workspace=workspace, rtol=1e-10).converged
    np.testing.assert_allclose(x.to_host(), truth, rtol=1e-7, atol=1e-8)


def test_unsupported_modes_layouts_aliases_and_workspaces_reject_before_writes(
    gpu, system
):
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
        gpu.prepare(
            gpu.CsrMatrix.from_host(cpu_csr(matrix)), dense_kernel="cublas"
        )
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
        gpu.solve_into(
            A,
            b,
            x,
            workspace=workspace,
            preconditioner="jacobi",
            algorithm="explicit",
        )
    np.testing.assert_array_equal(x.to_host(), np.zeros_like(rhs))


@pytest.mark.parametrize("kernel", ["native", "cublas"])
@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_true_norm_and_numerical_failure_statuses(gpu, kernel, execution_mode):
    A = gpu.DenseMatrix.from_host(np.eye(2))
    for magnitude in (1e-200, 1e200):
        b = gpu.Vector.from_host(np.array([magnitude, -magnitude]))
        limited = gpu.solve(
            A, b, dense_kernel=kernel, execution_mode=execution_mode, maxiter=0
        )
        assert limited.status == "max_iterations"
        assert math.isclose(
            limited.residual_norm,
            math.hypot(magnitude, magnitude),
            rel_tol=1e-14,
        )
        failed = gpu.solve(
            A, b, dense_kernel=kernel, execution_mode=execution_mode
        )
        assert failed.status == "non_finite"
        assert math.isclose(
            failed.residual_norm,
            math.hypot(magnitude, magnitude),
            rel_tol=1e-14,
        )
    for magnitude in (1e-150, 1e150):
        b = gpu.Vector.from_host(np.array([magnitude, -magnitude]))
        result = gpu.solve(
            A, b, dense_kernel=kernel, execution_mode=execution_mode
        )
        assert result.converged
        np.testing.assert_array_equal(result.x.to_host(), b.to_host())
    for bad in (np.nan, np.inf):
        invalid = gpu.DenseMatrix.from_host(np.array([[bad, 0.0], [0.0, 1.0]]))
        b = gpu.Vector.from_host([1.0, 1.0])
        assert (
            gpu.solve(
                invalid, b, dense_kernel=kernel, execution_mode=execution_mode
            ).status
            == "non_finite"
        )


def torch_gpu():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA PyTorch")
    return torch


def test_cuda_interface_rejects_host_pointers_and_ambiguous_producer_stream(
    gpu, system
):
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


@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_torch_zero_copy_strict_layout_and_borrowed_nondefault_stream(
    gpu, system, execution_mode
):
    torch = torch_gpu()
    matrix, rhs, truth = system
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        A = torch.tensor(matrix, dtype=torch.float64, device="cuda")
        b = torch.tensor(rhs, dtype=torch.float64, device="cuda")
        output = torch.zeros_like(b)
        plan = gpu.prepare(A, stream=stream.cuda_stream)
        workspace = plan.create_workspace(
            stream=stream.cuda_stream, execution_mode=execution_mode
        )
        current = torch.cuda.current_stream().cuda_stream
        info = plan.solve_into(b, output, workspace=workspace, rtol=1e-10)
        assert info.converged
        assert torch.cuda.current_stream().cuda_stream == current
        np.testing.assert_allclose(
            output.cpu().numpy(), truth, rtol=1e-7, atol=1e-8
        )
        for bad in (
            b.float(),
            torch.empty(b.size(0) * 2, device="cuda", dtype=torch.float64)[
                ::2
            ],
        ):
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
    assert plan.solve_into(
        Export(b, True), x, workspace=workspace, rtol=1e-10
    ).converged
    np.testing.assert_allclose(x.to_host(), truth, rtol=1e-7, atol=1e-8)
    with pytest.raises(ValueError, match="writable"):
        plan.solve_into(Export(b), Export(b, True), workspace=workspace)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_device_csr_snapshot_validates_canonical_indices(gpu, dtype):
    torch = torch_gpu()
    index_dtype = getattr(torch, dtype)
    pointers = torch.tensor([0, 2, 4], device="cuda", dtype=index_dtype)
    columns = torch.tensor([0, 1, 0, 1], device="cuda", dtype=index_dtype)
    values = torch.tensor(
        [4.0, 1.0, 1.0, 3.0], device="cuda", dtype=torch.float64
    )
    matrix = gpu.CsrMatrix.from_device(pointers, columns, values, (2, 2))
    assert matrix.nnz == 4
    values.fill_(np.nan)
    answer = gpu.solve(
        matrix, gpu.Vector.from_host([1.0, 2.0]), preconditioner="jacobi"
    )
    assert answer.converged
    np.testing.assert_allclose(answer.x.to_host(), [1 / 11, 7 / 11])
    for bad in ([1, 0, 0, 1], [0, 0, 0, 1], [0, 2, 0, 1]):
        with pytest.raises(ValueError):
            gpu.CsrMatrix.from_device(
                pointers,
                torch.tensor(bad, device="cuda", dtype=index_dtype),
                torch.ones(4, device="cuda", dtype=torch.float64),
                (2, 2),
            )


@pytest.mark.parametrize("kernel", ["native", "cublas"])
@pytest.mark.parametrize("execution_mode", ["host", "device"])
def test_shared_immutable_plan_with_independent_concurrent_workspaces(
    gpu, system, kernel, execution_mode
):
    torch = torch_gpu()
    matrix, rhs, truth = system
    plan = gpu.prepare_host(
        matrix, dense_kernel=kernel, preconditioner="jacobi"
    )
    b = gpu.Vector.from_host(rhs)
    streams = [torch.cuda.Stream() for _ in range(2)]

    def run(index):
        stream = streams[index]
        workspace = plan.create_workspace(
            stream=stream.cuda_stream, execution_mode=execution_mode
        )
        x = plan.create_vector(stream=stream.cuda_stream)
        assert plan.solve_into(b, x, workspace=workspace, rtol=1e-10).converged
        np.testing.assert_allclose(
            x.to_host(stream=stream.cuda_stream), truth, rtol=1e-7, atol=1e-8
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))


def test_execution_mode_is_fixed_per_workspace_and_never_inferred_from_the_plan(
    gpu,
):
    plan = gpu.prepare_host(np.eye(2))
    assert gpu.Workspace(2).execution_mode == "host"
    assert plan.create_workspace().execution_mode == "host"
    b, x = gpu.Vector.from_host([1.0, 2.0]), plan.create_vector()
    workspaces = [
        plan.create_workspace(execution_mode=mode)
        for mode in ("host", "device")
    ]
    for workspace in workspaces * 2:
        x.fill(0)
        assert plan.solve_into(b, x, workspace=workspace).converged
        np.testing.assert_array_equal(x.to_host(), [1.0, 2.0])
    for invalid in ("auto", "", "unknown"):
        with pytest.raises(ValueError, match="execution_mode"):
            gpu.Workspace(2, execution_mode=invalid)
        with pytest.raises(ValueError, match="execution_mode"):
            plan.create_workspace(execution_mode=invalid)
        with pytest.raises(ValueError, match="execution_mode"):
            gpu.solve(
                gpu.DenseMatrix.from_host(np.eye(2)), b, execution_mode=invalid
            )
    x.fill(0)
    assert plan.solve_into(b, x, workspace=workspaces[1]).converged
