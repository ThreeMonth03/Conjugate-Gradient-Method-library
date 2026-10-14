"""Exercise the same fixtures, adapters and collector used by make pyprof."""

import json
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
    assert load_artifact(replay)["cases"][0]["input_sha256"] == artifact["cases"][0]["input_sha256"]


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
