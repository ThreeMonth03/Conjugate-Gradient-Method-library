"""Exercise the same fixtures, adapters and collector used by make pyprof."""

import json
import math
import os
import subprocess
import sys
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np
import pytest
from cgbench.artifacts import (
    load_artifact,
    summarize,
    validate_artifact,
    write_artifact,
)
from cgbench.backends import Outcome, make_executor
from cgbench.cases import SUITES, CaseSpec, input_hash, make_case
from cgbench.collector import (
    check_outcome,
    collect_case,
    time_candidates,
    williams_rows,
)


@pytest.fixture
def solver_options():
    return {
        "algorithm": "standard",
        "rtol": 1e-8,
        "atol": 1e-12,
        "maxiter": 2000,
        "residual_refresh": 50,
        "fused": True,
    }


@pytest.fixture
def smoke_spec(solver_options):
    return {
        "suite": "smoke",
        "cases": [case.to_dict() for case in SUITES["smoke"]],
        "implementations": ["numpy-out"],
        "scope": "prepared",
        "threads": 1,
        "seed": 20261015,
        "sampling": {"warmups": 1, "repetitions": 2, "rounds": 2},
        "solver": solver_options,
    }


@pytest.mark.parametrize("count", range(2, 8))
def test_williams_balances_positions_and_predecessors(count):
    names = tuple(range(count))
    rows = williams_rows(names)
    assert len(rows) == count * (2 if count % 2 else 1)
    for row in rows:
        assert set(row) == set(names)
    for prefix_length in range(1, len(rows) + 1):
        for position in range(count):
            counts = Counter(row[position] for row in rows[:prefix_length])
            values = [counts[name] for name in names]
            assert max(values) - min(values) <= 1
    transitions = Counter(pair for row in rows for pair in zip(row, row[1:], strict=False))
    assert set(transitions) == set(permutations(names, 2))
    assert set(transitions.values()) == {2 if count % 2 else 1}


def test_times_blocks_not_schedule_or_warmup():
    calls = []
    ticks = iter(range(0, 10000, 100))
    executors = {name: lambda name=name: calls.append(name) for name in ("A", "B", "C")}
    sampling = {"warmups": 1, "repetitions": 3, "rounds": 6}
    rows = williams_rows(tuple(executors))
    orders, elapsed = time_candidates(executors, sampling, clock=lambda: next(ticks))
    assert calls[:3] == list(rows[-1])
    assert calls[3:] == [name for order in orders for name in order for _ in range(3)]
    assert elapsed == {name: [100] * 6 for name in executors}
    with pytest.raises(ValueError, match="complete Williams cycle"):
        time_candidates(executors, {**sampling, "rounds": 5})


def test_fixture_is_reproducible_spd():
    spec = SUITES["smoke"][0]
    first = make_case(spec, 17)
    second = make_case(spec, 17)
    assert input_hash(spec, first) == input_hash(spec, second)
    assert input_hash(spec, first) != input_hash(spec, make_case(spec, 18))
    matrix, rhs, initial, expected = first
    assert np.array_equal(matrix, matrix.T)
    np.linalg.cholesky(matrix)
    np.testing.assert_allclose(matrix @ expected, rhs, rtol=1e-14, atol=1e-14)
    assert not initial.any()


@pytest.mark.parametrize("scope", ("prepared", "api"))
@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
@pytest.mark.parametrize("refresh", (0, 3, 50))
def test_numpy_adapter_resets_and_converges(solver_options, scope, algorithm, refresh):
    arrays = make_case(SUITES["smoke"][0])[:3]
    options = {**solver_options, "algorithm": algorithm, "residual_refresh": refresh}
    executor = make_executor("numpy-out", arrays, options, scope, 1)
    first = executor()
    iterations, matvecs = first.iterations, first.matvecs
    assert check_outcome(first, arrays, options)["eligible"]
    expected = first.x.copy()
    first.x.fill(1e6)
    second = executor()
    assert check_outcome(second, arrays, options)["eligible"]
    assert second.iterations == iterations
    assert second.matvecs == matvecs
    np.testing.assert_array_equal(second.x, expected)
    assert not arrays[2].any()


def test_true_residual_gate_rejects_finite_wrong_solution(solver_options):
    arrays = make_case(CaseSpec("wrong", 8))[:3]
    outcome = Outcome(np.zeros(8), "converged", 0, 0, 0.0)
    correctness = check_outcome(outcome, arrays, solver_options)
    assert not correctness["eligible"]
    assert correctness["true_residual_norm"] > correctness["residual_threshold"]


@pytest.mark.parametrize("magnitude", (1e-200, 1e200))
def test_true_residual_gate_preserves_extreme_float64_norms(solver_options, magnitude):
    rhs = np.array([magnitude, -magnitude])
    arrays = (np.eye(2), rhs, np.zeros(2))
    outcome = Outcome(np.zeros(2), "converged", 0, 1, 0.0)
    correctness = check_outcome(outcome, arrays, {**solver_options, "atol": 0.0})
    assert not correctness["eligible"]
    assert correctness["true_residual_norm"] == math.hypot(magnitude, -magnitude)
    assert correctness["true_residual_norm"] > correctness["residual_threshold"]


@pytest.mark.parametrize("magnitude", (1e-200, 1e200))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
@pytest.mark.parametrize("implementation", ("numpy-out", "torch-eager-out"))
def test_reference_reports_unrepresentable_inner_product_range(
    solver_options, magnitude, preconditioner, implementation
):
    if implementation == "torch-eager-out":
        pytest.importorskip("torch")
    rhs = np.array([magnitude, -magnitude])
    arrays = (np.eye(2), rhs, np.zeros(2))
    options = {**solver_options, "atol": 0.0, "preconditioner": preconditioner}
    outcome = make_executor(implementation, arrays, options, "prepared", 1)()
    assert outcome.status == "non_finite" and not outcome.converged
    assert outcome.residual_norm == math.hypot(magnitude, -magnitude)
    assert not check_outcome(outcome, arrays, options)["eligible"]


@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
def test_maxiter_status_and_final_true_residual(solver_options, algorithm):
    arrays = make_case(SUITES["smoke"][0])[:3]
    options = {**solver_options, "maxiter": 1, "algorithm": algorithm}
    result = make_executor("numpy-out", arrays, options, "prepared", 1)()
    assert result.status == "max_iterations"
    assert result.iterations == 1
    assert result.matvecs >= 3
    np.testing.assert_allclose(
        result.residual_norm, np.linalg.norm(arrays[1] - arrays[0] @ result.x)
    )


def test_zero_rhs_and_initial_convergence(solver_options):
    matrix, rhs, initial = make_case(SUITES["smoke"][0])[:3]
    rhs.fill(0)
    result = make_executor("numpy-out", (matrix, rhs, initial), solver_options, "prepared", 1)()
    assert result.converged and result.iterations == 0 and result.matvecs == 1


@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
def test_zero_iterations_uses_only_initial_true_residual(solver_options, algorithm):
    arrays = make_case(SUITES["smoke"][0])[:3]
    options = {**solver_options, "maxiter": 0, "algorithm": algorithm}
    result = make_executor("numpy-out", arrays, options, "prepared", 1)()
    assert result.status == "max_iterations"
    assert result.iterations == 0 and result.matvecs == 1


def test_replay_refuses_changed_inputs_or_candidates_before_timing(smoke_spec):
    case_spec = SUITES["smoke"][0]
    spec = {**smoke_spec, "expected_hashes": {case_spec.name: "wrong"}}
    with pytest.raises(ValueError, match="input hash mismatch"):
        collect_case(case_spec, spec)
    spec = {**smoke_spec, "expected_candidates": {case_spec.name: ["native"]}}
    with pytest.raises(ValueError, match="eligible candidates changed"):
        collect_case(case_spec, spec)
    spec = {**smoke_spec, "expected_round_orders": {case_spec.name: [["wrong"]]}}
    with pytest.raises(ValueError, match="timing schedule changed"):
        collect_case(case_spec, spec)


def test_collector_artifact_roundtrip_and_raw_stats(smoke_spec, tmp_path):
    case = collect_case(SUITES["smoke"][0], smoke_spec)
    artifact = {
        "schema_version": 1,
        "spec": smoke_spec,
        "environment": {},
        "cases": [case],
    }
    path = tmp_path / "benchmark.json"
    write_artifact(artifact, path)
    restored = load_artifact(path)
    assert restored == artifact
    entry = restored["cases"][0]["results"][0]
    entry["round_elapsed_ns"] = [100, 300]
    assert summarize(restored)[0]["median_ns"] == 100
    entry["round_elapsed_ns"] = [100]
    with pytest.raises(ValueError, match="raw nanosecond"):
        validate_artifact(restored)


def test_launcher_sets_child_thread_environment_and_replays(tmp_path):
    output = tmp_path / "first.json"
    command = [
        sys.executable,
        "-m",
        "cgbench",
        "run",
        "--suite",
        "smoke",
        "--implementations",
        "numpy-out",
        "--no-fused",
        "--threads",
        "1",
        "--warmups",
        "0",
        "--rounds",
        "2",
        "--repetitions",
        "1",
        "--output",
        str(output),
    ]
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "python") + os.pathsep + env.get("PYTHONPATH", "")
    before = os.environ.get("OMP_NUM_THREADS")
    subprocess.run(command, env=env, check=True, capture_output=True, text=True)
    artifact = json.loads(output.read_text())
    assert artifact["spec"]["solver"]["fused"] is False
    assert all(value == "1" for value in artifact["environment"]["thread_environment"].values())
    assert os.environ.get("OMP_NUM_THREADS") == before
    replay = tmp_path / "replay.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "cgbench",
            "replay",
            str(output),
            "--output",
            str(replay),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    replayed = load_artifact(replay)
    assert replayed["cases"][0]["input_sha256"] == artifact["cases"][0]["input_sha256"]
    assert replayed["spec"] == artifact["spec"]
    assert replayed["cases"][0]["round_orders"] == artifact["cases"][0]["round_orders"]


@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
def test_native_and_numpy_use_shared_fixture(solver_options, algorithm):
    pytest.importorskip("cgpy")
    arrays = make_case(SUITES["smoke"][0])[:3]
    options = {**solver_options, "algorithm": algorithm}
    native = make_executor("native", arrays, options, "prepared", 1)()
    reference = make_executor("numpy-out", arrays, options, "prepared", 1)()
    assert check_outcome(native, arrays, options)["eligible"]
    assert check_outcome(reference, arrays, options)["eligible"]
    np.testing.assert_allclose(native.x, reference.x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("scope", ("prepared", "api"))
@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
def test_torch_reference_is_optional_and_correct(solver_options, scope, algorithm):
    pytest.importorskip("torch")
    arrays = make_case(SUITES["smoke"][0])[:3]
    solver_options = {**solver_options, "algorithm": algorithm}
    executor = make_executor("torch-eager-out", arrays, solver_options, scope, 1)
    first = executor()
    assert check_outcome(first, arrays, solver_options)["eligible"]
    iterations = first.iterations
    first.x.fill(1e6)
    second = executor()
    assert check_outcome(second, arrays, solver_options)["eligible"]
    assert second.iterations == iterations


def test_csr_fixture_is_canonical_reproducible_and_spd():
    pytest.importorskip("scipy")
    spec = SUITES["smoke-csr"][0]
    first = make_case(spec, 17)
    assert input_hash(spec, first) == input_hash(spec, make_case(spec, 17))
    assert input_hash(spec, first) != input_hash(spec, make_case(spec, 18))
    matrix, rhs, initial, expected = first
    assert matrix.format == "csr" and matrix.has_canonical_format
    assert matrix.nnz == 5 * spec.size - 4 * spec.grid
    assert (matrix != matrix.T).nnz == 0
    # Densification is restricted to this tiny independent SPD check.
    np.linalg.cholesky(matrix.toarray())
    np.testing.assert_allclose(matrix @ expected, rhs)
    assert not initial.any()
    original_hash = input_hash(spec, first)
    matrix.data[0] += 0.1
    assert input_hash(spec, first) != original_hash


@pytest.mark.parametrize("implementation", ("scipy-out", "torch-eager-out"))
@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
@pytest.mark.parametrize("scope", ("prepared", "api"))
def test_csr_adapters_share_fixture_and_reset(solver_options, implementation, algorithm, scope):
    pytest.importorskip("scipy")
    if implementation == "torch-eager-out":
        pytest.importorskip("torch")
    arrays = make_case(SUITES["smoke-csr"][0])[:3]
    options = {**solver_options, "algorithm": algorithm, "residual_refresh": 5}
    executor = make_executor(implementation, arrays, options, scope, 1)
    first = executor()
    assert check_outcome(first, arrays, options)["eligible"]
    expected = first.x.copy()
    iterations, matvecs = first.iterations, first.matvecs
    first.x.fill(1e6)
    second = executor()
    assert check_outcome(second, arrays, options)["eligible"]
    assert second.iterations == iterations and second.matvecs == matvecs
    np.testing.assert_array_equal(second.x, expected)


def test_csr_collection_does_not_densify(solver_options, smoke_spec, monkeypatch):
    sparse = pytest.importorskip("scipy.sparse")

    def forbidden(*args, **kwargs):
        raise AssertionError("benchmark attempted to densify CSR")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    spec = {**smoke_spec, "implementations": ["scipy-out"]}
    case = collect_case(SUITES["smoke-csr"][0], spec)
    assert case["layout"] == "CSR" and case["nnz"] == 64
    assert case["results"][0]["status"] == "measured"


@pytest.mark.parametrize("algorithm", ("standard", "explicit"))
def test_native_csr_uses_shared_fixture(solver_options, algorithm):
    pytest.importorskip("scipy")
    cgpy = pytest.importorskip("cgpy")
    if not hasattr(cgpy, "CSRMatrix"):
        pytest.skip("native CSR binding is not available")
    arrays = make_case(SUITES["smoke-csr"][0])[:3]
    options = {**solver_options, "algorithm": algorithm}
    native = make_executor("native", arrays, options, "prepared", 1)()
    reference = make_executor("scipy-out", arrays, options, "prepared", 1)()
    assert check_outcome(native, arrays, options)["eligible"]
    assert check_outcome(reference, arrays, options)["eligible"]
    np.testing.assert_allclose(native.x, reference.x, rtol=1e-6, atol=1e-8)


@pytest.mark.parametrize("scope", ("prepared", "api"))
@pytest.mark.parametrize("implementation", ("scipy-reference-pcg", "torch-pcg", "scipy-pcg"))
def test_jacobi_pcg_on_scaled_poisson(solver_options, scope, implementation):
    pytest.importorskip("scipy")
    if implementation == "torch-pcg":
        pytest.importorskip("torch")
    arrays = make_case(SUITES["smoke-sparse-solvers"][0])[:3]
    options = {**solver_options, "residual_refresh": 5}
    executor = make_executor(implementation, arrays, options, scope, 1)
    first = executor()
    assert check_outcome(first, arrays, options)["eligible"]
    expected = first.x.copy()
    iterations = first.iterations
    assert executor.benchmark_metadata()["preconditioner_stored_bytes"] == arrays[1].nbytes
    first.x.fill(1e6)
    second = executor()
    assert check_outcome(second, arrays, options)["eligible"]
    assert second.iterations == iterations
    np.testing.assert_array_equal(second.x, expected)
    matched_cg = "scipy-cg" if implementation == "scipy-pcg" else "scipy-reference-cg"
    cg = make_executor(matched_cg, arrays, options, scope, 1)()
    assert first.iterations < cg.iterations


@pytest.mark.parametrize("scope", ("prepared", "api"))
@pytest.mark.parametrize("implementation", ("scipy-lu-total", "scipy-lu-reuse"))
def test_sparse_direct_setup_memory_and_true_residual(solver_options, scope, implementation):
    pytest.importorskip("scipy")
    arrays = make_case(SUITES["smoke-sparse-solvers"][1])[:3]
    executor = make_executor(implementation, arrays, solver_options, scope, 1)
    first = executor()
    assert check_outcome(first, arrays, solver_options)["eligible"]
    metadata = executor.benchmark_metadata()
    assert metadata["factor_stored_bytes"] > 0
    assert metadata["factorization_setup_elapsed_ns"] >= 0
    assert metadata["factorization_in_timed_call"] == implementation.endswith("total")
    if implementation == "scipy-lu-total":
        assert executor.last_factor is None
    expected = first.x.copy()
    first.x.fill(1e6)
    second = executor()
    assert check_outcome(second, arrays, solver_options)["eligible"]
    np.testing.assert_array_equal(second.x, expected)
    assert second.iterations == 0 and second.matvecs == 1


@pytest.mark.parametrize("implementation", ("scipy-cg", "scipy-pcg"))
def test_official_scipy_tiny_residual_respects_absolute_tolerance(solver_options, implementation):
    sparse = pytest.importorskip("scipy.sparse")
    rhs = np.array([1e-200, -1e-200])
    arrays = (2.0 * sparse.eye(2, format="csr"), rhs, np.zeros(2))
    options = {**solver_options, "atol": 1e-199}
    outcome = make_executor(implementation, arrays, options, "prepared", 1)()
    assert outcome.converged
    assert check_outcome(outcome, arrays, options)["eligible"]
    assert not check_outcome(outcome, arrays, {**options, "atol": 0.0})["eligible"]


def test_sparse_solver_collection_uses_same_inputs_and_separate_algorithm_tags(smoke_spec):
    pytest.importorskip("scipy")
    spec = {
        **smoke_spec,
        "comparison": "sparse-solvers",
        "implementations": ["scipy-cg", "scipy-pcg", "scipy-lu-total", "scipy-lu-reuse"],
    }
    case = collect_case(SUITES["smoke-sparse-solvers"][0], spec)
    assert case["cycle_rows"] == 4 and case["sampling"]["rounds"] == 4
    assert all(entry["status"] == "measured" for entry in case["results"])
    assert all(
        entry["setup"]["implementation_role"] == "algorithm_comparison" for entry in case["results"]
    )
    assert case["results"][0]["setup"]["library_solver"] == "scipy.sparse.linalg.cg"
    assert case["input_stored_bytes"] > 0


def test_reference_pcg_rejects_explicit_algorithm(solver_options):
    pytest.importorskip("scipy")
    arrays = make_case(SUITES["smoke-sparse-solvers"][0])[:3]
    with pytest.raises(ValueError, match="requires the standard recurrence"):
        make_executor(
            "scipy-reference-pcg",
            arrays,
            {**solver_options, "algorithm": "explicit"},
            "prepared",
            1,
        )


def test_sparse_solver_smoke_all_six_candidates(smoke_spec):
    pytest.importorskip("scipy")
    cgpy = pytest.importorskip("cgpy")
    if not hasattr(cgpy, "CSRMatrix"):
        pytest.skip("native CSR binding is not available")
    from cgbench.cli import SPARSE_SOLVERS

    spec = {**smoke_spec, "comparison": "sparse-solvers", "implementations": list(SPARSE_SOLVERS)}
    for fixture in SUITES["smoke-sparse-solvers"]:
        case = collect_case(fixture, spec)
        assert case["cycle_rows"] == 6 and case["sampling"]["rounds"] == 6
        assert all(entry["status"] == "measured" for entry in case["results"])
