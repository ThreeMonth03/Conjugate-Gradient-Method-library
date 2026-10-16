"""Count actual direction restarts and preserve unknown historical evidence."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from cgbench import artifacts, collector, execution, validation
from cgbench.adapters import cuda as cuda_adapters
from cgbench.cases import CaseSpec
from cgbench.reference.engine import ReferenceExecutor
from cgbench.reference.numpy import NumpyOps

from scripts import compare_performance, run_paired_performance


@pytest.fixture
def options():
    return {
        "algorithm": "standard",
        "rtol": 0.0,
        "atol": 0.0,
        "maxiter": 3,
        "residual_refresh": 1,
        "preconditioner": "none",
        "fused": True,
    }


def test_outcome_defaults_to_unknown_instead_of_inventing_zero_restarts():
    legacy = execution.Outcome(object(), "converged", 4, 6, 0.0)
    assert legacy.restarts is None
    assert legacy.converged
    current = execution.Outcome(object(), "converged", 4, 6, 0.0, restarts=2)
    assert current.restarts == 2


@pytest.mark.parametrize("scope", ("api", "prepared"))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
@pytest.mark.parametrize("fused", (False, True))
def test_native_and_reference_forced_refresh_counts_reset_each_call(
    options, scope, preconditioner, fused
):
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    initial = np.zeros_like(rhs)
    arrays = matrix, rhs, initial
    supplied = {**options, "preconditioner": preconditioner, "fused": fused}
    native = execution.make_executor("native", arrays, supplied, scope, 1)
    reference = execution.make_executor(
        "numpy-out", arrays, supplied, scope, 1
    )

    for _ in range(2):
        actual, expected = native(), reference()
        for outcome in (actual, expected):
            assert outcome.status == "max_iterations"
            assert (outcome.iterations, outcome.matvecs, outcome.restarts) == (
                3,
                7,
                2,
            )
            assert outcome.residual_norm == pytest.approx(
                np.linalg.norm(rhs - matrix @ outcome.x)
            )
        np.testing.assert_allclose(
            actual.x, expected.x, rtol=1e-13, atol=1e-14
        )
        # A prepared output is reused: the next invocation must reset both
        # numerical state and the restart count rather than accumulating work.
        actual.x.fill(1e6)
        expected.x.fill(-1e6)
    np.testing.assert_array_equal(initial, np.zeros(2))


@pytest.mark.parametrize("implementation", ("native", "numpy-out"))
@pytest.mark.parametrize("scope", ("api", "prepared"))
@pytest.mark.parametrize(
    "scenario",
    (
        "initial",
        "zero_limit",
        "last_iteration",
        "converged_refresh",
        "curvature",
        "explicit",
    ),
)
def test_residual_checks_and_early_exits_do_not_count_as_restarts(
    options, implementation, scope, scenario
):
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    initial = np.zeros(2)
    expected_status = "max_iterations"
    supplied = dict(options)
    if scenario == "initial":
        initial[:] = 1.0
        rhs = matrix @ initial
        expected_status = "converged"
    elif scenario == "zero_limit":
        supplied["maxiter"] = 0
    elif scenario == "last_iteration":
        supplied["maxiter"] = 1
    elif scenario == "converged_refresh":
        matrix = np.eye(2)
        expected_status = "converged"
    elif scenario == "curvature":
        matrix = -np.eye(2)
        expected_status = "non_positive_curvature"
    else:
        supplied["algorithm"] = "explicit"
        expected_status = None
    outcome = execution.make_executor(
        implementation, (matrix, rhs, initial), supplied, scope, 1
    )()
    if expected_status is not None:
        assert outcome.status == expected_status
    assert outcome.restarts == 0
    if scenario in ("initial", "zero_limit"):
        assert (outcome.iterations, outcome.matvecs) == (0, 1)
    elif scenario in ("last_iteration", "converged_refresh"):
        assert (outcome.iterations, outcome.matvecs) == (1, 3)


@pytest.mark.parametrize("scope", ("api", "prepared", "session"))
@pytest.mark.parametrize("failed_value", (0.0, float("nan")))
def test_reference_pcg_failure_before_replacing_direction_is_not_a_restart(
    options, scope, failed_value
):
    class FailedRestartOps(NumpyOps):
        def __init__(self, arrays):
            super().__init__(arrays)
            self.precondition_calls = 0
            self.direction_initializations = 0

        def precondition(self, inverse, residual, out):
            self.precondition_calls += 1
            super().precondition(inverse, residual, out)
            if self.precondition_calls == 2:
                # The refreshed residual is valid, but M*r fails before the
                # search direction can actually be replaced.
                out.fill(failed_value)

        def initialize_direction(self, residual, direction, explicit):
            self.direction_initializations += 1
            super().initialize_direction(residual, direction, explicit)

    arrays = (
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        np.array([1.0, 2.0]),
        np.zeros(2),
    )
    ops = FailedRestartOps(arrays)
    outcome = ReferenceExecutor(
        ops, {**options, "preconditioner": "jacobi"}, scope
    )()
    assert outcome.status == "non_finite"
    assert (outcome.iterations, outcome.matvecs, outcome.restarts) == (1, 3, 0)
    assert ops.precondition_calls == 2
    assert ops.direction_initializations == 1
    assert np.isfinite(outcome.residual_norm)


@pytest.mark.parametrize("scope", ("api", "prepared", "session"))
@pytest.mark.parametrize("mode", ("host", "device"))
def test_cuda_adapter_and_host_gate_preserve_restart_counts_without_cuda(
    options, scope, mode
):
    downloads = []
    calls = []
    host_x = object()

    class Vector:
        def copy_from_device(self, source, **kwargs):
            calls.append(("reset", source, kwargs))

        def to_host(self, **kwargs):
            downloads.append(kwargs)
            return host_x

    x = Vector()
    info = SimpleNamespace(
        x=x,
        status="converged",
        iterations=4,
        matvecs=9,
        residual_norm=1e-12,
        restarts=2,
    )

    def solve(*args, **kwargs):
        calls.append(("solve", args, kwargs))
        return info

    execute = object.__new__(cuda_adapters.CudaNativeExecutor)
    execute.scope = scope
    execute.device = 2
    execute.kernel = "native"
    execute.execution_mode = mode
    execute.options = {
        key: value for key, value in options.items() if key != "preconditioner"
    }
    execute.preconditioner = "none"
    execute.matrix, execute.rhs, execute.initial = object(), object(), object()
    execute.x = x
    execute.workspace = object()
    execute.cuda = SimpleNamespace(solve=solve)
    execute.plan = SimpleNamespace(solve_into=solve)
    execute.session = SimpleNamespace(solve=solve)

    outcome = execute()
    assert outcome.x is x and outcome.restarts == 2
    assert downloads == []
    assert len(calls) == (2 if scope == "prepared" else 1)
    host = execute.host_outcome(outcome)
    assert host.x is host_x and host.restarts == 2
    assert (
        host.status,
        host.iterations,
        host.matvecs,
        host.residual_norm,
    ) == (
        "converged",
        4,
        9,
        1e-12,
    )
    assert downloads == [{"stream": 0}]


@pytest.mark.parametrize("restarts", (None, 0, 2))
def test_torch_host_conversion_preserves_known_and_unknown_restarts(restarts):
    resident_x = object()
    host_x = np.array([1.0, 2.0])
    exports = []

    def export(value):
        exports.append(value)
        return host_x

    execute = object.__new__(cuda_adapters.TorchCudaExecutor)
    execute.ops = SimpleNamespace(export=export)
    outcome = execution.Outcome(
        resident_x, "converged", 4, 9, 1e-12, restarts=restarts
    )
    host = execute.host_outcome(outcome)
    assert exports == [resident_x]
    assert host.restarts is restarts
    np.testing.assert_array_equal(host.x, host_x)
    assert not np.shares_memory(host.x, host_x)


def measured_result():
    return {
        "name": "native",
        "status": "measured",
        "solver_status": "converged",
        "iterations": 4,
        "matvecs": 9,
        "true_residual_norm": 1e-12,
        "residual_threshold": 1e-8,
        "round_elapsed_ns": [100] * 4,
    }


@pytest.mark.parametrize("restarts", (None, 0, 2))
def test_true_residual_gate_exports_restart_evidence(options, restarts):
    arrays = np.eye(2), np.ones(2), np.zeros(2)
    outcome = execution.Outcome(
        np.ones(2), "converged", 4, 9, 0.0, restarts=restarts
    )
    checked = collector.check_outcome(outcome, arrays, options)
    assert checked["eligible"]
    assert checked["restarts"] == restarts


@pytest.mark.parametrize("restarts", (True, 1.5, 5))
def test_true_residual_gate_rejects_invalid_counts_before_admitting_timing(
    options, restarts
):
    arrays = np.eye(2), np.ones(2), np.zeros(2)
    outcome = execution.Outcome(
        np.ones(2), "converged", 4, 9, 0.0, restarts=restarts
    )
    checked = collector.check_outcome(outcome, arrays, options)
    assert not checked["eligible"]
    assert "restart" in checked["reason"]


@pytest.mark.parametrize("after_restarts", (1, None))
def test_collector_discards_samples_when_post_measurement_restarts_change(
    options, monkeypatch, after_restarts
):
    from cgbench import cases

    fixture = np.eye(2), np.ones(2), np.zeros(2), np.ones(2)
    outcomes = iter(
        execution.Outcome(np.ones(2), "converged", 4, 9, 0.0, restarts=count)
        for count in (2, after_restarts)
    )

    class Executor:
        def __call__(self):
            return next(outcomes)

        def benchmark_metadata(self):
            return {}

    monkeypatch.setattr(cases, "make_case", lambda *args: fixture)
    monkeypatch.setattr(cases, "input_hash", lambda *args: "a" * 64)
    monkeypatch.setattr(execution, "make_executor", lambda *a, **k: Executor())
    monkeypatch.setattr(
        collector,
        "time_candidates",
        lambda *args: ([["native"]] * 4, {"native": [100] * 4}),
    )
    spec = {
        "seed": 17,
        "implementations": ["native"],
        "solver": options,
        "scope": "prepared",
        "threads": 1,
        "sampling": {"rounds": 4, "repetitions": 1, "warmups": 0},
    }
    result = collector.collect_case(CaseSpec("reset", 2), spec)["results"][0]
    assert result["status"] == "invalid"
    assert result["round_elapsed_ns"] == []
    assert "post-measurement numerical work differs" in result["reason"]


@pytest.mark.parametrize(
    "restarts", (-1, True, False, 1.5, "1", float("nan"), 5)
)
def test_invalid_known_restart_counts_are_rejected(restarts):
    result = {**measured_result(), "restarts": restarts}
    with pytest.raises(ValueError, match="restart|operation counts"):
        validation.validate_converged_result(result)


@pytest.mark.parametrize("restarts", (None, 0, 2, 4))
def test_valid_optional_restart_counts_do_not_change_legacy_validation(
    restarts,
):
    result = measured_result()
    validation.validate_converged_result(result)
    result["restarts"] = restarts
    validation.validate_converged_result(result)


def benchmark_artifact():
    case = {"name": "dense-4", "size": 4, "condition_scale": 100.0}
    sampling = {"rounds": 4, "repetitions": 1, "warmups": 0}
    return {
        "schema_version": 1,
        "metric": "time_to_solution",
        "spec": {
            "scope": "prepared",
            "threads": 1,
            "seed": 17,
            "solver": {"algorithm": "standard", "rtol": 1e-8},
            "implementations": ["native"],
            "cases": [case],
            "sampling": sampling,
        },
        "cases": [
            {
                "case": case,
                "input_sha256": "a" * 64,
                "sampling": sampling,
                "cycle_rows": 1,
                "round_orders": [["native"] for _ in range(4)],
                "results": [measured_result()],
            }
        ],
    }


@pytest.mark.parametrize("representation", ("missing", "null", "known"))
def test_schema_one_roundtrip_preserves_restart_evidence_without_backfill(
    tmp_path, representation
):
    data = benchmark_artifact()
    entry = data["cases"][0]["results"][0]
    if representation != "missing":
        entry["restarts"] = None if representation == "null" else 2
    original = copy.deepcopy(data)
    path = tmp_path / "restarts.json"
    artifacts.write_artifact(data, path)
    assert data == original
    restored = artifacts.load_artifact(path)
    assert restored == original and restored["schema_version"] == 1
    loaded_entry = restored["cases"][0]["results"][0]
    assert ("restarts" in loaded_entry) == (representation != "missing")
    assert loaded_entry.get("restarts") == entry.get("restarts")


def save_comparison_blocks(tmp_path, prefix, restart_values):
    paths = []
    for index, restarts in enumerate(restart_values):
        data = benchmark_artifact()
        if restarts != "missing":
            data["cases"][0]["results"][0]["restarts"] = restarts
        path = tmp_path / f"{prefix}-{index}.json"
        path.write_text(json.dumps(data), encoding="utf8")
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    "old,new",
    ((2, 1), (2, "missing"), ("missing", 2), (2, None), (None, 2)),
)
def test_strict_comparator_rejects_restart_mismatch_or_one_unknown_side(
    tmp_path, old, new
):
    baseline = save_comparison_blocks(tmp_path, "old", [old, old])
    current = save_comparison_blocks(tmp_path, "new", [new, new])
    with pytest.raises(ValueError, match="restart|numerical work"):
        compare_performance.compare(baseline, current, resamples=100)


def test_strict_comparator_checks_restart_counts_within_each_stage(tmp_path):
    baseline = save_comparison_blocks(tmp_path, "old", [2, 1])
    current = save_comparison_blocks(tmp_path, "new", [2, 2])
    with pytest.raises(ValueError, match="restart|numerical work"):
        compare_performance.compare(baseline, current, resamples=100)


@pytest.mark.parametrize("restarts", (0, 2, "missing", None))
def test_strict_report_discloses_whether_restart_parity_was_verified(
    tmp_path, restarts
):
    baseline = save_comparison_blocks(tmp_path, "old", [restarts] * 2)
    current = save_comparison_blocks(tmp_path, "new", [restarts] * 2)
    report = compare_performance.compare(baseline, current, resamples=100)
    known = type(restarts) is int
    assert report["verdict"] == "supported_no_material_regression"
    assert report["restart_parity_verified"] is known
    row = report["cases"][0]
    assert row["restarts"] == (restarts if known else None)
    assert row["restart_parity_verified"] is known
    if not known:
        assert any(
            "restart parity is unverified" in note.lower()
            for note in report["limitations"]
        )


def test_legacy_missing_and_explicit_null_share_the_unknown_contract(tmp_path):
    baseline = save_comparison_blocks(tmp_path, "old", ["missing", None])
    current = save_comparison_blocks(tmp_path, "new", [None, "missing"])
    report = compare_performance.compare(baseline, current, resamples=100)
    assert report["restart_parity_verified"] is False
    assert report["cases"][0]["restarts"] is None


@pytest.mark.parametrize("restarts", (-1, True, 1.5, 5))
def test_invalid_restart_evidence_is_rejected_by_saved_artifact_consumers(
    tmp_path, restarts
):
    data = benchmark_artifact()
    data["cases"][0]["results"][0]["restarts"] = restarts
    with pytest.raises(ValueError, match="restart|operation counts"):
        artifacts.validate_artifact(data)
    baseline = save_comparison_blocks(tmp_path, "old", [restarts] * 2)
    current = save_comparison_blocks(tmp_path, "new", [restarts] * 2)
    with pytest.raises(ValueError, match="restart|operation counts"):
        compare_performance.compare(baseline, current, resamples=100)


def paired_job(restarts="missing"):
    job = {
        "case": {"name": "fixture"},
        "input_sha256": "a" * 64,
        "iterations": 4,
        "matvecs": 9,
        "residual_threshold": 1e-8,
    }
    if restarts != "missing":
        job["restarts"] = restarts
    return job


def worker_response(restarts="missing"):
    return {
        **paired_job(restarts),
        "solver_status": "converged",
        "eligible": True,
        "true_residual_norm": 1e-12,
    }


@pytest.mark.parametrize(
    "old,new",
    ((2, 1), (2, "missing"), ("missing", 2), (2, None), (None, 2)),
)
@pytest.mark.parametrize("require_hash", (False, True))
def test_paired_worker_protocol_rejects_different_or_unknown_restart_work(
    old, new, require_hash
):
    response = worker_response(new)
    if not require_hash:
        # Timed sample replies omit the untimed true-residual gate fields.
        response.pop("eligible")
    with pytest.raises(ValueError, match="restart|numerical work"):
        run_paired_performance._verify_work(
            paired_job(old), response, require_hash=require_hash
        )


@pytest.mark.parametrize(
    "old,new",
    ((2, 2), ("missing", "missing"), ("missing", None), (None, "missing")),
)
def test_paired_worker_protocol_accepts_equal_known_or_both_unknown_counts(
    old, new
):
    run_paired_performance._verify_work(
        paired_job(old), worker_response(new), require_hash=True
    )


@pytest.mark.parametrize("restarts", (-1, False, 1.5, 5))
def test_timed_paired_samples_cannot_hide_invalid_restart_counts(restarts):
    response = worker_response(restarts)
    response.pop("eligible")
    # False compares equal to zero in Python; validation must precede equality.
    with pytest.raises(ValueError, match="restart|operation counts"):
        run_paired_performance._verify_work(paired_job(0), response)


@pytest.mark.parametrize("restarts", (2, "missing"))
def test_paired_preparation_retains_restart_evidence(restarts):
    class Worker:
        def request(self, operation, **kwargs):
            assert operation == "prepare"
            return worker_response(restarts)

    workers = {name: Worker() for name in ("baseline", "current")}
    expected, preparation = run_paired_performance._prepare_job(
        workers, {"case": {"name": "fixture"}}
    )
    assert expected.get("restarts") == (2 if restarts == 2 else None)
    assert expected["restart_parity_verified"] is (restarts == 2)
    assert preparation["baseline"] == worker_response(restarts)
    assert preparation["current"] == worker_response(restarts)


def test_paired_round_rejects_changed_restarts_before_returning_samples():
    class Worker:
        def __init__(self, restarts):
            self.restarts = restarts

        def request(self, operation, **kwargs):
            assert operation == "sample"
            return {
                "elapsed_ns": 700,
                "iterations": 4,
                "matvecs": 9,
                "restarts": self.restarts,
                "solver_status": "converged",
            }

    workers = {"baseline": Worker(2), "current": Worker(1)}
    with pytest.raises(ValueError, match="restart|numerical work"):
        run_paired_performance._paired_round(workers, paired_job(2), 0, 7)


@pytest.mark.parametrize("restarts", (2, "missing"))
def test_paired_report_discloses_known_or_unverified_restart_parity(
    tmp_path, monkeypatch, restarts
):
    class Worker:
        def __init__(self, name):
            self.name = name

        def request(self, operation, **kwargs):
            if operation in ("prepare", "verify"):
                return worker_response(restarts)
            if operation == "metadata":
                return {"native": {"module_sha256": self.name}}
            assert operation == "sample"
            response = worker_response(restarts)
            response.pop("eligible")
            response["elapsed_ns"] = 1000 * kwargs["repetitions"]
            return response

        def close(self):
            pass

    job = {
        "case": {"name": "fixture"},
        "suite": "smoke",
        "scope": "prepared",
        "implementation": "native",
    }
    monkeypatch.setattr(run_paired_performance, "Worker", Worker)
    monkeypatch.setattr(run_paired_performance, "_jobs", lambda spec: [job])
    output = tmp_path / "paired-restarts.json"
    assert (
        run_paired_performance.main(
            [
                "--baseline-pythonpath",
                "baseline",
                "--current-pythonpath",
                "current",
                "--output",
                str(output),
                "--rounds",
                "50",
                "--block-length",
                "5",
                "--bootstrap-samples",
                "100",
            ]
        )
        == 0
    )
    report = json.loads(output.read_text(encoding="utf8"))
    known = restarts == 2
    assert report["restart_parity_verified"] is known
    assert report["cases"][0]["restart_parity_verified"] is known
    assert report["cases"][0]["restarts"] == (2 if known else None)
    if not known:
        assert any(
            "restart parity is unverified" in note.lower()
            for note in report["limitations"]
        )
