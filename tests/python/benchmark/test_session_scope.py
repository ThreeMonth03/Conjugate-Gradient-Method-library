"""CUDA session timing contracts, testable without a CUDA runtime."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
from cgbench import artifacts, cli, collector, cuda_runtime, execution
from cgbench.adapters import cuda as cuda_adapters
from cgbench.reference.engine import ReferenceExecutor
from cgbench.reference.numpy import NumpyOps


@pytest.fixture
def options():
    return {
        "algorithm": "standard",
        "rtol": 1e-10,
        "atol": 1e-12,
        "maxiter": 100,
        "residual_refresh": 5,
        "fused": True,
        "preconditioner": "none",
    }


def session_artifact(options, mode):
    names = ["cuda-native", "cuda-cublas", "torch-cuda-eager"]
    orders = [list(row) for row in collector.williams_rows(names)]
    case = {"name": "dense-16", "size": 16, "condition_scale": 100.0}
    sampling = {"rounds": len(orders), "repetitions": 1, "warmups": 0}
    spec = {
        "comparison": "implementations",
        "device": "cuda",
        "cuda_device": 2,
        "scope": "session",
        "suite": "smoke-cuda-dense",
        "threads": 1,
        "seed": 17,
        "implementations": names,
        "cases": [case],
        "solver": {**options, "execution_mode": mode},
        "sampling": sampling,
    }
    return {
        "schema_version": 1,
        "metric": "time_to_solution",
        "spec": spec,
        "environment": {"cuda": {"device": 2}},
        "cases": [
            {
                "case": case,
                "dtype": "float64",
                "input_sha256": "a" * 64,
                "sampling": sampling,
                "cycle_rows": len(orders),
                "schedule": "williams",
                "round_orders": orders,
                "results": [
                    {
                        "name": name,
                        "status": "measured",
                        "solver_status": "converged",
                        "true_residual_norm": 1e-12,
                        "residual_threshold": 1e-8,
                        "iterations": 4,
                        "matvecs": 6,
                        "round_elapsed_ns": [100] * len(orders),
                    }
                    for name in names
                ],
            }
        ],
    }


@pytest.mark.parametrize("mode", ("host", "device"))
def test_session_cli_preserves_cuda_policy_and_default_scope(
    monkeypatch, mode
):
    captured = []
    monkeypatch.setattr(
        cli, "_launch", lambda spec, output: captured.append(spec) or 0
    )
    arguments = [
        "run",
        "--device",
        "cuda",
        "--cuda-device",
        "2",
        "--suite",
        "smoke-cuda-dense",
        "--implementations",
        "cuda-native,cuda-cublas,torch-cuda-eager",
        "--cuda-execution-mode",
        mode,
    ]
    assert cli.main([*arguments, "--scope", "session"]) == 0
    assert captured[-1]["scope"] == "session"
    assert captured[-1]["cuda_device"] == 2
    assert captured[-1]["solver"].get("execution_mode", "host") == mode
    assert cli.main(arguments) == 0
    assert captured[-1]["scope"] == "prepared"


def test_cpu_session_cli_fails_before_worker_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("CPU session request reached a worker")

    monkeypatch.setattr(cli, "_launch", forbidden)
    with pytest.raises(SystemExit, match="session scope requires CUDA"):
        cli.main(["run", "--scope", "session"])


@pytest.mark.parametrize(
    "name", ("native", "native-blas", "numpy-out", "torch-eager-out")
)
def test_cpu_session_factory_rejects_before_constructing_an_adapter(
    options, name
):
    # No fixture exists: validation must precede native/library setup.
    with pytest.raises(ValueError, match="session scope requires CUDA"):
        execution.make_executor(name, None, options, "session", 1)


@pytest.mark.parametrize("name", execution.CUDA_IMPLEMENTATIONS)
def test_cuda_session_factory_dispatches_explicit_kernel_and_device(
    options, monkeypatch, name
):
    calls = []
    expected = object()

    def native(arrays, supplied, scope, device, kernel):
        calls.append((arrays, supplied, scope, device, kernel))
        return expected

    def torch(arrays, supplied, scope, threads, device):
        calls.append((arrays, supplied, scope, device, threads))
        return expected

    monkeypatch.setattr(cuda_adapters, "CudaNativeExecutor", native)
    monkeypatch.setattr(cuda_adapters, "TorchCudaExecutor", torch)
    arrays = object()
    supplied = {**options, "execution_mode": "device"}
    before = supplied.copy()
    actual = execution.make_executor(
        name, arrays, supplied, "session", 1, device="cuda", cuda_device=2
    )
    assert actual is expected and supplied == before
    selector = (
        1
        if name == "torch-cuda-eager"
        else ("cublas" if name == "cuda-cublas" else "native")
    )
    assert calls == [(arrays, supplied, "session", 2, selector)]


@pytest.mark.parametrize("mode", ("host", "device"))
def test_session_artifact_replay_preserves_scope_inputs_and_schedule(
    options, tmp_path, monkeypatch, mode
):
    original = session_artifact(options, mode)
    path = tmp_path / "session.json"
    artifacts.write_artifact(original, path)
    assert artifacts.load_artifact(path) == original
    captured = []
    monkeypatch.setattr(
        cli, "_launch", lambda spec, output: captured.append(spec) or 0
    )
    assert (
        cli.main(
            ["replay", str(path), "--output", str(tmp_path / "replayed.json")]
        )
        == 0
    )
    replay = captured.pop()
    assert {key: replay[key] for key in original["spec"]} == original["spec"]
    case = original["cases"][0]
    assert replay["expected_hashes"] == {"dense-16": case["input_sha256"]}
    assert replay["expected_candidates"] == {
        "dense-16": original["spec"]["implementations"]
    }
    assert replay["resolved_sampling"] == {"dense-16": case["sampling"]}
    assert replay["expected_round_orders"] == {
        "dense-16": case["round_orders"]
    }

    invalid = copy.deepcopy(original)
    invalid["spec"]["device"] = "cpu"
    invalid["spec"]["solver"]["execution_mode"] = "host"
    with pytest.raises(ValueError, match="session scope requires CUDA"):
        artifacts.validate_artifact(invalid)


@pytest.mark.parametrize("mode", ("host", "device"))
def test_session_worker_preflights_cuda_before_fixture_collection_or_writes(
    options, tmp_path, monkeypatch, mode
):
    spec = session_artifact(options, mode)["spec"]
    calls = []

    def unavailable(names, device):
        calls.append((names, device))
        raise RuntimeError("requested CUDA device unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "unavailable session reached collection or output"
        )

    monkeypatch.setattr(cuda_runtime, "preflight_cuda", unavailable)
    monkeypatch.setattr(collector, "collect_case", forbidden)
    monkeypatch.setattr(artifacts, "environment_metadata", forbidden)
    monkeypatch.setattr(artifacts, "write_artifact", forbidden)
    output = tmp_path / "unavailable.json"
    with pytest.raises(RuntimeError, match="CUDA device unavailable"):
        cli._run_worker(spec, output)
    assert calls == [(spec["implementations"], 2)]
    assert not output.exists()


@pytest.mark.parametrize(
    "algorithm,preconditioner",
    (("standard", "none"), ("standard", "jacobi"), ("explicit", "none")),
)
def test_reference_session_reuses_only_workvectors_and_observes_matrix_changes(
    options, algorithm, preconditioner
):
    class ObservedOps(NumpyOps):
        def __init__(self, arrays):
            self.allocated = []
            self.inverses = []
            self.workspace_calls = 0
            super().__init__(arrays)

        def empty_vector(self):
            result = super().empty_vector()
            self.allocated.append(result)
            return result

        def workspace(self):
            self.workspace_calls += 1
            return super().workspace()

        def snapshot_matrix(self):
            raise AssertionError("session must read the current matrix")

        def inverse_diagonal(self):
            result = super().inverse_diagonal()
            self.inverses.append(result)
            return result

    matrix = np.diag([2.0, 3.0, 5.0])
    rhs = np.array([2.0, 6.0, 15.0])
    initial = np.zeros(3)
    ops = ObservedOps((matrix, rhs, initial))
    execute = ReferenceExecutor(
        ops,
        {**options, "algorithm": algorithm, "preconditioner": preconditioner},
        "session",
    )
    workvectors = tuple(ops.allocated)
    assert len(workvectors) == (5 if preconditioner == "jacobi" else 4)
    assert ops.matrix is matrix and ops.workspace_calls == 0
    probes = 1 if preconditioner == "jacobi" else 0
    assert len(ops.inverses) == probes
    first = execute()
    assert first.converged and ops.workspace_calls == 0
    np.testing.assert_allclose(
        first.x, np.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-12
    )
    saved_first = first.x.copy()
    assert len(ops.allocated) == len(workvectors) + 1
    assert ops.allocated[-1] is first.x
    assert (
        len(ops.inverses) == 2 * probes
    )  # Untimed setup probe plus first call.

    matrix[:] = [[4.0, 0.25, 0.0], [0.25, 7.0, -0.5], [0.0, -0.5, 9.0]]
    second = execute()
    assert second.converged and ops.matrix is matrix
    np.testing.assert_allclose(
        second.x, np.linalg.solve(matrix, rhs), rtol=1e-10, atol=1e-12
    )
    np.testing.assert_array_equal(first.x, saved_first)
    np.testing.assert_array_equal(initial, np.zeros(3))
    assert not np.shares_memory(first.x, second.x)
    assert all(
        not np.shares_memory(second.x, vector) for vector in workvectors
    )
    assert all(
        ops.allocated[index] is vector
        for index, vector in enumerate(workvectors)
    )
    assert (
        len(ops.allocated) == len(workvectors) + 2 and ops.workspace_calls == 0
    )
    assert len(ops.inverses) == 3 * probes
    if preconditioner == "jacobi":
        assert ops.inverses[-1] is not ops.inverses[-2]
        np.testing.assert_array_equal(ops.inverses[-1], 1 / np.diag(matrix))
    assert execute.benchmark_metadata()["setup_in_timed_call"] is True


@pytest.mark.parametrize("mode", ("host", "device"))
@pytest.mark.parametrize("kernel", ("native", "cublas"))
def test_native_session_adapter_reuses_session_but_delegates_fresh_calls(
    options, monkeypatch, mode, kernel
):
    counts = {"upload": 0, "download": 0, "prepare": 0, "session": 0}
    calls = []

    class Vector:
        def __init__(self, value):
            self.value = np.array(value, dtype=np.float64, copy=True)

        @classmethod
        def from_host(cls, value, **kwargs):
            counts["upload"] += 1
            return cls(value)

        def to_host(self, **kwargs):
            counts["download"] += 1
            return self.value.copy()

    class Matrix:
        @classmethod
        def from_host(cls, value, **kwargs):
            counts["upload"] += 1
            return SimpleNamespace(value=np.array(value, copy=True))

    class Session:
        def __init__(self, size, **kwargs):
            counts["session"] += 1
            assert size == 2
            assert kwargs == {
                "device": 2,
                "stream": 0,
                "dense_kernel": kernel,
                "execution_mode": mode,
            }

        def solve(self, matrix, rhs, *, x0, preconditioner, **kwargs):
            calls.append((matrix, rhs, x0, preconditioner, kwargs))
            np.testing.assert_array_equal(x0.value, [0.25, -0.5])
            return SimpleNamespace(
                x=Vector(np.linalg.solve(matrix.value, rhs.value)),
                status="converged",
                iterations=2,
                matvecs=4,
                residual_norm=0.0,
                restarts=0,
            )

    def prepare(*args, **kwargs):
        counts["prepare"] += 1
        return SimpleNamespace(
            preconditioner_setup_ns=3, preconditioner_nbytes=16
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("session called the one-shot solver")

    cuda = SimpleNamespace(
        DenseMatrix=Matrix,
        Vector=Vector,
        Session=Session,
        prepare=prepare,
        solve=forbidden,
        synchronize=lambda **kwargs: None,
    )
    monkeypatch.setattr(cuda_runtime, "native_cuda", lambda device: cuda)
    arrays = (
        np.diag([2.0, 3.0]),
        np.array([2.0, 6.0]),
        np.array([0.25, -0.5]),
    )
    supplied = {**options, "execution_mode": mode, "preconditioner": "jacobi"}
    execute = cuda_adapters.CudaNativeExecutor(
        arrays, supplied, "session", 2, kernel
    )
    assert counts == {"upload": 3, "download": 0, "prepare": 1, "session": 1}
    first = execute()
    saved_first = first.x.value.copy()
    execute.matrix.value[:] = [[4.0, 0.5], [0.5, 5.0]]
    second = execute()
    assert len(calls) == 2 and counts["session"] == counts["prepare"] == 1
    assert counts["upload"] == 3 and counts["download"] == 0
    for matrix, rhs, x0, preconditioner, kwargs in calls:
        assert (
            matrix is execute.matrix
            and rhs is execute.rhs
            and x0 is execute.initial
        )
        assert preconditioner == "jacobi"
        assert kwargs == {
            key: value
            for key, value in options.items()
            if key != "preconditioner"
        }
    np.testing.assert_array_equal(first.x.value, saved_first)
    assert not np.shares_memory(first.x.value, second.x.value)
    host = execute.host_outcome(second)
    assert counts["download"] == 1
    np.testing.assert_allclose(
        host.x, np.linalg.solve(execute.matrix.value, arrays[1])
    )
    metadata = execute.benchmark_metadata()
    assert metadata["workspace_reused"] is True
    assert metadata["output_reused"] is False
    assert metadata["preconditioner_reused"] is False
    assert metadata["setup_in_timed_call"] is True
