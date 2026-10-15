"""CUDA benchmark contracts run without a GPU; hardware solves are explicit skips."""

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from cgbench.adapters.cuda import CudaNativeExecutor
from cgbench.artifacts import load_artifact, validate_artifact, write_artifact
from cgbench.cases import SUITES, input_hash, make_case
from cgbench.cli import main
from cgbench.collector import (
    _check_executor_outcome,
    check_outcome,
    collect_case,
    time_candidates,
)
from cgbench.cuda_runtime import preflight_cuda
from cgbench.execution import make_executor
from cgbench.reference.torch import TorchOps
from cgbench.reference.torch_cuda import TorchCudaOps


@pytest.fixture
def options():
    return {
        "algorithm": "standard",
        "rtol": 1e-8,
        "atol": 1e-12,
        "maxiter": 2000,
        "residual_refresh": 5,
        "fused": True,
        "preconditioner": "none",
    }


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--device", "cuda"], "requires explicit"),
        (["--cuda-execution-mode", "device"], "requires CUDA"),
        (
            ["--device", "cuda", "--implementations", "numpy-out"],
            "CPU fallback",
        ),
        (["--implementations", "cuda-native"], "require --device cuda"),
        (["--suite", "smoke-cuda-dense"], "require --device cuda"),
        (
            [
                "--device",
                "cuda",
                "--suite",
                "smoke-csr",
                "--implementations",
                "cuda-cublas",
            ],
            "requires a dense suite",
        ),
    ],
)
def test_cuda_cli_rejects_invalid_modes_before_launch(
    arguments, message, monkeypatch
):
    def forbidden(*args):
        raise AssertionError("invalid mode reached the launcher")

    monkeypatch.setattr("cgbench.cli._launch", forbidden)
    with pytest.raises(SystemExit, match=message):
        main(["run", *arguments])


def test_cuda_cli_records_explicit_policy_without_changing_cpu_defaults(
    monkeypatch,
):
    captured = []
    monkeypatch.setattr(
        "cgbench.cli._launch", lambda spec, output: captured.append(spec) or 0
    )
    assert main(["run"]) == 0
    assert captured[-1]["implementations"] == [
        "native",
        "numpy-out",
        "torch-eager-out",
    ]
    assert "device" not in captured[-1]
    assert (
        main(
            [
                "run",
                "--device",
                "cuda",
                "--cuda-device",
                "2",
                "--suite",
                "smoke-cuda-dense",
                "--implementations",
                "cuda-native,cuda-cublas,torch-cuda-eager",
                "--no-fused",
            ]
        )
        == 0
    )
    assert (
        captured[-1]["device"] == "cuda" and captured[-1]["cuda_device"] == 2
    )
    assert captured[-1]["solver"]["fused"] is False
    assert captured[-1]["comparison"] == "implementations"
    assert "execution_mode" not in captured[-1]["solver"]
    host_spec = copy.deepcopy(captured[-1])
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
        "--no-fused",
        "--cuda-execution-mode",
        "device",
    ]
    assert main(arguments) == 0
    assert captured[-1] == {
        **host_spec,
        "solver": {**host_spec["solver"], "execution_mode": "device"},
    }


@pytest.mark.parametrize(
    "cuda_name,cpu_name",
    [
        ("smoke-cuda-dense", "smoke"),
        ("smoke-cuda-csr", "smoke-csr"),
        ("cuda-dense-f64-v1", "cpu-dense-f64-v1"),
        ("cuda-csr-f64-v1", "cpu-csr-f64-v1"),
    ],
)
def test_cuda_fixture_aliases_preserve_shared_spec_and_hash(
    cuda_name, cpu_name
):
    assert SUITES[cuda_name] == SUITES[cpu_name]
    # Hash a small actual fixture, without GPU imports or large matrix creation.
    if "smoke" in cpu_name:
        if "csr" in cpu_name:
            pytest.importorskip("scipy")
        spec = SUITES[cuda_name][0]
        assert input_hash(spec, make_case(spec, 17)) == input_hash(
            SUITES[cpu_name][0], make_case(SUITES[cpu_name][0], 17)
        )


def test_gpu_fences_bracket_completed_blocks_not_individual_repetitions():
    events = []

    class AsynchronousExecutor:
        def synchronize(self):
            events.append("fence")

        def __call__(self):
            events.append("solve")

    ticks = iter((100, 190))

    def clock():
        events.append("clock")
        return next(ticks)

    orders, elapsed = time_candidates(
        {"gpu": AsynchronousExecutor()},
        {"warmups": 1, "rounds": 1, "repetitions": 3},
        clock=clock,
    )
    assert events == [
        "solve",
        "fence",
        "clock",
        "solve",
        "solve",
        "solve",
        "fence",
        "clock",
    ]
    assert orders == [["gpu"]] and elapsed == {"gpu": [90]}


def test_explicit_cuda_unavailability_fails_without_cpu_fallback(monkeypatch):
    from cgbench import cuda_runtime

    def absent(device):
        raise RuntimeError(
            f"CUDA requested but native CUDA device {device} is unavailable"
        )

    monkeypatch.setattr(cuda_runtime, "native_cuda", absent)
    with pytest.raises(RuntimeError, match="CUDA requested.*unavailable"):
        preflight_cuda(["cuda-native"], 0)
    with pytest.raises(ValueError, match="explicit CUDA"):
        preflight_cuda(["numpy-out"], 0)
    with pytest.raises(ValueError, match="nonnegative"):
        preflight_cuda(["cuda-native"], -1)


@pytest.mark.parametrize("scope", ("prepared", "api"))
@pytest.mark.parametrize("execution_mode", ("host", "device"))
def test_native_adapter_charges_device_reset_and_keeps_transfers_at_gates(
    options, scope, execution_mode, monkeypatch
):
    if execution_mode == "device":
        options = {**options, "execution_mode": execution_mode}
    arrays = make_case(SUITES["smoke"][0])[:3]
    expected = np.linalg.solve(arrays[0], arrays[1])
    counts = {
        name: 0
        for name in ("upload", "download", "reset", "prepared", "one_shot")
    }

    class Vector:
        def __init__(self, value):
            self.value = np.array(value, copy=True)

        @classmethod
        def from_host(cls, value, **kwargs):
            assert kwargs["device"] == 2
            counts["upload"] += 1
            return cls(value)

        def copy_from_device(self, other, **kwargs):
            counts["reset"] += 1
            np.copyto(self.value, other.value)

        def to_host(self, **kwargs):
            counts["download"] += 1
            return self.value.copy()

    class Plan:
        preconditioner_setup_ns = 13
        preconditioner_nbytes = 0

        def create_workspace(self, **kwargs):
            assert kwargs["execution_mode"] == execution_mode
            return object()

        def create_vector(self, **kwargs):
            return Vector(np.empty_like(expected))

        def solve_into(self, b, x, **kwargs):
            assert "execution_mode" not in kwargs
            counts["prepared"] += 1
            np.testing.assert_array_equal(x.value, arrays[2])
            x.value[:] = expected
            return SimpleNamespace(
                status="converged",
                iterations=16,
                matvecs=20,
                residual_norm=0.0,
            )

    def solve(matrix, b, x0, **kwargs):
        assert kwargs["device"] == 2
        assert kwargs["execution_mode"] == execution_mode
        counts["one_shot"] += 1
        result = Vector(np.empty_like(expected))
        result.copy_from_device(x0)
        result.value[:] = expected
        return SimpleNamespace(
            x=result,
            status="converged",
            iterations=16,
            matvecs=20,
            residual_norm=0.0,
        )

    def prepare(matrix, **kwargs):
        assert kwargs["device"] == 2
        return Plan()

    cuda = SimpleNamespace(
        Vector=Vector,
        DenseMatrix=SimpleNamespace(from_host=lambda *a, **k: object()),
        synchronize=lambda **kwargs: None,
        prepare=prepare,
        solve=solve,
    )
    monkeypatch.setattr(
        "cgbench.cuda_runtime.native_cuda", lambda device: cuda
    )
    executor = CudaNativeExecutor(arrays, options, scope, 2, "native")
    assert executor.benchmark_metadata()["execution_mode"] == execution_mode
    assert "execution_mode" not in executor.options
    assert counts["upload"] == 2
    for _ in range(2):
        outcome = executor()
        assert counts["download"] == 0
        assert np.array_equal(outcome.x.value, expected)
        outcome.x.value.fill(1e6)
    assert counts["reset"] == 2
    assert counts["one_shot"] == (2 if scope == "api" else 0)
    assert counts["prepared"] == (2 if scope == "prepared" else 0)
    outcome = executor()
    assert _check_executor_outcome(executor, outcome, arrays, options)[
        "eligible"
    ]
    assert counts["download"] == 1 and counts["upload"] == 2
    assert executor.benchmark_metadata()["setup_in_timed_call"] == (
        scope == "api"
    )


def test_cuda_artifact_roundtrip_and_invalid_device_schema(tmp_path, options):
    spec = {
        "device": "cuda",
        "cuda_device": 0,
        "scope": "prepared",
        "threads": 1,
        "implementations": ["cuda-native"],
        "solver": options,
    }
    case = {
        "case": {"name": "dense-16"},
        "sampling": {"rounds": 1, "repetitions": 1},
        "cycle_rows": 1,
        "round_orders": [["cuda-native"]],
        "results": [
            {
                "name": "cuda-native",
                "status": "measured",
                "solver_status": "converged",
                "true_residual_norm": 1e-10,
                "residual_threshold": 1e-8,
                "iterations": 4,
                "matvecs": 6,
                "round_elapsed_ns": [100],
            }
        ],
    }
    data = {
        "schema_version": 1,
        "spec": spec,
        "environment": {"cuda": {"device": 0}},
        "cases": [case],
    }
    path = tmp_path / "gpu.json"
    write_artifact(data, path)
    assert load_artifact(path) == data
    for value in (-1, True, "0"):
        bad = copy.deepcopy(data)
        bad["spec"]["cuda_device"] = value
        with pytest.raises(ValueError, match="nonnegative integer"):
            validate_artifact(bad)
    for mode in (None, "auto", True):
        bad = copy.deepcopy(data)
        bad["spec"]["solver"]["execution_mode"] = mode
        with pytest.raises(ValueError, match="execution_mode"):
            validate_artifact(bad)
    for mode in ("host", "device"):
        good = copy.deepcopy(data)
        good["spec"]["solver"]["execution_mode"] = mode
        assert validate_artifact(good) is good
    bad = copy.deepcopy(data)
    bad["spec"]["device"] = "cpu"
    bad["spec"]["solver"]["execution_mode"] = "device"
    with pytest.raises(ValueError, match="requires CUDA"):
        validate_artifact(bad)
    bad = copy.deepcopy(data)
    bad["spec"]["implementations"] = ["numpy-out"]
    with pytest.raises(ValueError, match="cannot contain CPU"):
        validate_artifact(bad)


def test_cuda_metadata_hashes_only_the_loaded_native_extension(
    tmp_path, monkeypatch
):
    from cgbench import cuda_metadata

    binary = tmp_path / "_cuda.so"
    binary.write_bytes(b"actual loaded CUDA extension")
    monkeypatch.setitem(
        sys.modules, "cgpy._cuda", SimpleNamespace(__file__=str(binary))
    )
    monkeypatch.setitem(
        sys.modules,
        "cgpy.cuda",
        SimpleNamespace(
            build_info=lambda: {"toolkit": "12.8"},
            runtime_info=lambda **k: {"name": "GPU", **k},
        ),
    )
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(cuda_metadata.shutil, "which", lambda name: None)

    def absent(*args, **kwargs):
        raise FileNotFoundError("inventory unavailable")

    monkeypatch.setattr(cuda_metadata.subprocess, "run", absent)
    result = cuda_metadata.cuda_metadata(2)
    assert (
        result["native"]["module_sha256"]
        == hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    assert result["native"]["runtime_info"] == {"name": "GPU", "device": 2}
    assert result["torch"] is None


@pytest.mark.parametrize("ops_type", (TorchOps, TorchCudaOps))
@pytest.mark.parametrize(
    "diagonal", ([4.0, 3.0], [0.0, 3.0], [float("inf"), 3.0], [1e-320, 3.0])
)
def test_torch_jacobi_shares_validation_without_mutating_the_matrix(
    ops_type, diagonal
):
    torch = pytest.importorskip("torch")
    # The diagonal operations themselves support CPU tensors, so their shared
    # contract is testable without creating a CUDA context or changing devices.
    original = np.diag(diagonal)
    ops = ops_type.__new__(ops_type)
    ops.torch = torch
    ops.original_matrix = original.copy()
    ops.matrix = torch.from_numpy(original.copy())
    ops.sparse = False
    if diagonal[0] == 4.0:
        inverse = ops.inverse_diagonal()
        torch.testing.assert_close(
            inverse, torch.tensor([0.25, 1 / 3], dtype=torch.float64)
        )
        assert inverse.device == ops.matrix.device
    else:
        with pytest.raises(
            ValueError, match="finite and positive|finite positive"
        ):
            ops.inverse_diagonal()
    np.testing.assert_array_equal(ops.original_matrix, original)
    np.testing.assert_array_equal(ops.matrix.numpy(), original)


def test_torch_cuda_csr_diagonal_uses_the_prepared_device_positions():
    torch = pytest.importorskip("torch")
    ops = TorchCudaOps.__new__(TorchCudaOps)
    ops.torch = torch
    ops.sparse = True
    ops.matrix = torch.sparse_csr_tensor(
        torch.tensor([0, 2, 4]),
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([4.0, 1.0, 1.0, 3.0], dtype=torch.float64),
        size=(2, 2),
    )
    ops.diagonal_positions = torch.tensor([0, 3])
    inverse = ops.inverse_diagonal()
    torch.testing.assert_close(
        inverse, torch.tensor([0.25, 1 / 3], dtype=torch.float64)
    )
    torch.testing.assert_close(
        ops.matrix.values(),
        torch.tensor([4.0, 1.0, 1.0, 3.0], dtype=torch.float64),
    )


def _require_gpu(implementation):
    if implementation == "torch-cuda-eager":
        torch = pytest.importorskip("torch")
        if not torch.cuda.is_available():
            pytest.skip("CUDA PyTorch/device unavailable")
    else:
        from cgbench.cuda_runtime import native_cuda

        try:
            native_cuda(0)
        except (RuntimeError, ImportError):
            pytest.skip("native CUDA extension/device unavailable")


@pytest.mark.parametrize(
    "implementation", ("cuda-native", "cuda-cublas", "torch-cuda-eager")
)
@pytest.mark.parametrize("scope", ("prepared", "api", "session"))
@pytest.mark.parametrize(
    "algorithm,preconditioner",
    (("standard", "none"), ("standard", "jacobi"), ("explicit", "none")),
)
@pytest.mark.parametrize("execution_mode", ("host", "device"))
def test_actual_cuda_dense_resident_modes_reset_and_converge(
    options, implementation, scope, algorithm, preconditioner, execution_mode
):
    _require_gpu(implementation)
    arrays = make_case(SUITES["smoke"][0])[:3]
    options = {
        **options,
        "algorithm": algorithm,
        "preconditioner": preconditioner,
        "execution_mode": execution_mode,
    }
    executor = make_executor(
        implementation, arrays, options, scope, 1, device="cuda"
    )
    first = executor()
    host = executor.host_outcome(first)
    assert check_outcome(host, arrays, options)["eligible"]
    if implementation == "torch-cuda-eager":
        with executor.ops.torch.inference_mode():
            first.x.fill_(1e6)
    else:
        first.x.fill(1e6)
    second = executor.host_outcome(executor())
    if scope == "session":
        np.testing.assert_array_equal(
            executor.host_outcome(first).x, np.full_like(arrays[1], 1e6)
        )
    assert check_outcome(second, arrays, options)["eligible"]
    np.testing.assert_allclose(second.x, host.x, rtol=1e-10, atol=1e-12)
    assert (
        second.iterations == host.iterations and second.matvecs == host.matvecs
    )


@pytest.mark.parametrize("implementation", ("cuda-native", "torch-cuda-eager"))
@pytest.mark.parametrize("scope", ("prepared", "api", "session"))
@pytest.mark.parametrize("execution_mode", ("host", "device"))
def test_actual_cuda_csr_pcg_shares_cpu_fixture(
    options, implementation, scope, execution_mode
):
    _require_gpu(implementation)
    pytest.importorskip("scipy")
    arrays = make_case(SUITES["smoke-csr"][0])[:3]
    options = {
        **options,
        "preconditioner": "jacobi",
        "execution_mode": execution_mode,
    }
    executor = make_executor(
        implementation, arrays, options, scope, 1, device="cuda"
    )
    assert check_outcome(executor.host_outcome(executor()), arrays, options)[
        "eligible"
    ]


@pytest.mark.parametrize("execution_mode", ("host", "device"))
def test_actual_cuda_collector_and_replay_preserve_input_and_schedule(
    options, tmp_path, execution_mode
):
    _require_gpu("cuda-native")
    _require_gpu("torch-cuda-eager")
    if execution_mode == "device":
        options = {**options, "execution_mode": execution_mode}
    spec = {
        "device": "cuda",
        "cuda_device": 0,
        "scope": "prepared",
        "threads": 1,
        "suite": "smoke-cuda-dense",
        "seed": 17,
        "solver": options,
        "implementations": ["cuda-native", "cuda-cublas", "torch-cuda-eager"],
        "sampling": {"rounds": 6, "repetitions": 1, "warmups": 0},
    }
    fixture = SUITES["smoke-cuda-dense"][0]
    first = collect_case(fixture, spec)
    assert all(result["status"] == "measured" for result in first["results"])
    replay = {
        **spec,
        "expected_hashes": {fixture.name: first["input_sha256"]},
        "expected_candidates": {fixture.name: spec["implementations"]},
        "resolved_sampling": {fixture.name: first["sampling"]},
        "expected_round_orders": {fixture.name: first["round_orders"]},
    }
    second = collect_case(fixture, replay)
    assert second["input_sha256"] == first["input_sha256"]
    assert second["round_orders"] == first["round_orders"]


@pytest.mark.parametrize(
    "implementations",
    ("torch-cuda-eager", "cuda-native,cuda-cublas,torch-cuda-eager"),
)
def test_actual_torch_cuda_launcher_metadata_and_fresh_process_replay(
    tmp_path, implementations
):
    _require_gpu("torch-cuda-eager")
    if "cuda-native" in implementations:
        _require_gpu("cuda-native")
    first = tmp_path / "cuda-first.json"
    replay = tmp_path / "cuda-replay.json"
    assert (
        main(
            [
                "run",
                "--device",
                "cuda",
                "--suite",
                "smoke-cuda-dense",
                "--implementations",
                implementations,
                "--seed",
                "17",
                "--warmups",
                "0",
                "--rounds",
                "2",
                "--repetitions",
                "1",
                "--output",
                str(first),
            ]
        )
        == 0
    )
    artifact = load_artifact(first)
    gpu = artifact["environment"]["cuda"]
    assert gpu["device"] == 0
    assert gpu["torch"]["runtime_version"] is not None
    assert len(gpu["torch"]["compute_capability"]) == 2
    assert gpu["torch"]["matmul_allow_tf32"] is False
    if "cuda-native" in implementations:
        binary = Path(gpu["native"]["module_path"])
        assert (
            gpu["native"]["module_sha256"]
            == hashlib.sha256(binary.read_bytes()).hexdigest()
        )
        assert gpu["native"]["build_info"] and gpu["native"]["runtime_info"]
    assert main(["replay", str(first), "--output", str(replay)]) == 0
    replayed = load_artifact(replay)
    assert replayed["spec"] == artifact["spec"]
    assert (
        replayed["cases"][0]["input_sha256"]
        == artifact["cases"][0]["input_sha256"]
    )
    assert (
        replayed["cases"][0]["round_orders"]
        == artifact["cases"][0]["round_orders"]
    )


def test_torch_reference_keeps_its_control_when_native_device_mode_is_selected(
    options, monkeypatch
):
    from cgbench.adapters import cuda as adapters
    from cgbench.reference import engine

    supplied = {**options, "execution_mode": "device"}
    captured = {}
    ops = SimpleNamespace(
        device=SimpleNamespace(index=0),
        upload_ns=3,
        torch=SimpleNamespace(
            cuda=SimpleNamespace(
                current_stream=lambda device: SimpleNamespace(cuda_stream=0)
            )
        ),
    )

    def initialize(self, selected_ops, selected_options, scope):
        self.ops = selected_ops
        captured.update(selected_options)

    monkeypatch.setattr(adapters.torch_cuda, "TorchCudaOps", lambda *a: ops)
    monkeypatch.setattr(engine.ReferenceExecutor, "__init__", initialize)
    monkeypatch.setattr(
        engine.ReferenceExecutor, "benchmark_metadata", lambda self: {}
    )
    monkeypatch.setattr(
        adapters.TorchCudaExecutor, "synchronize", lambda self: None
    )
    reference = adapters.TorchCudaExecutor(None, supplied, "prepared", 1, 0)
    assert captured == options
    assert supplied["execution_mode"] == "device"
    metadata = reference.benchmark_metadata()
    assert metadata["native_execution_mode"] == "device"
    assert metadata["native_execution_mode_applies"] is False


def test_cpu_executor_rejects_device_control_before_constructing_any_backend(
    options,
):
    with pytest.raises(ValueError, match="requires CUDA"):
        make_executor(
            "native",
            None,
            {**options, "execution_mode": "device"},
            "prepared",
            1,
        )
