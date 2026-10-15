"""Reject invalid numerical evidence and preserve reports on write failure."""

import json
import sys

import pytest
from cgbench import artifacts, json_io

from scripts import compare_performance, run_paired_performance


@pytest.fixture
def artifact():
    case = {"name": "dense-4", "size": 4}
    return {
        "schema_version": 1,
        "metric": "time_to_solution",
        "spec": {
            "scope": "prepared",
            "implementations": ["native"],
            "cases": [case],
        },
        "cases": [
            {
                "case": case,
                "input_sha256": "a" * 64,
                "sampling": {"rounds": 1, "repetitions": 1},
                "cycle_rows": 1,
                "round_orders": [["native"]],
                "results": [
                    {
                        "name": "native",
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
        ],
    }


@pytest.mark.parametrize(
    "field,value",
    [
        ("solver_status", "max_iterations"),
        ("true_residual_norm", 1e-4),
        ("true_residual_norm", float("nan")),
        ("true_residual_norm", None),
        ("residual_threshold", float("inf")),
        ("residual_threshold", -1.0),
        ("iterations", True),
        ("matvecs", -1),
    ],
)
def test_measured_results_must_pass_the_same_numerical_gate(
    artifact, field, value, tmp_path
):
    artifact["cases"][0]["results"][0][field] = value
    with pytest.raises(ValueError):
        artifacts.validate_artifact(artifact)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(artifact))
    with pytest.raises(ValueError):
        compare_performance._read_artifact(path)


def test_rounding_scale_residual_difference_remains_eligible(
    artifact, tmp_path
):
    entry = artifact["cases"][0]["results"][0]
    entry["true_residual_norm"] = entry["residual_threshold"] * (
        1 + 32 * sys.float_info.epsilon
    )
    path = tmp_path / "rounding.json"
    artifacts.write_artifact(artifact, path)
    assert artifacts.load_artifact(path) == artifact
    assert compare_performance._read_artifact(path).measurements


def test_failed_optional_candidates_remain_valid_artifacts(artifact):
    case = artifact["cases"][0]
    artifact["spec"]["implementations"] = ["native-blas"]
    case["round_orders"] = [[]]
    case["results"] = [
        {
            "name": "native-blas",
            "status": "ineligible",
            "reason": "optional library unavailable",
            "round_elapsed_ns": [],
        }
    ]
    assert artifacts.validate_artifact(artifact) is artifact


@pytest.mark.parametrize("failure", ["serialize", "write", "fsync", "replace"])
def test_atomic_write_preserves_existing_report_and_removes_temporary_files(
    failure, tmp_path, monkeypatch
):
    path = tmp_path / "report.json"
    previous = '{"previous": true}\n'
    path.write_text(previous)
    data = {"current": True}
    if failure == "serialize":
        data["invalid"] = float("nan")
        error = ValueError
    else:
        error = OSError

        def fail(*args, **kwargs):
            if failure == "write":
                args[1].write('{"partial":')
            raise OSError(f"injected {failure} failure")

        if failure == "write":
            monkeypatch.setattr(json_io.json, "dump", fail)
        else:
            monkeypatch.setattr(json_io.os, failure, fail)
    with pytest.raises(error):
        json_io.write_json(data, path)
    assert path.read_text() == previous
    assert list(tmp_path.iterdir()) == [path]


def test_atomic_writer_creates_a_complete_report_without_schema_assumptions(
    tmp_path,
):
    path = tmp_path / "new" / "report.json"
    report = {"title": "量測摘要", "cases": []}
    json_io.write_json(report, path)
    assert json.loads(path.read_text()) == report
    assert path.read_bytes().endswith(b"\n")
    assert list(path.parent.iterdir()) == [path]


def test_artifact_and_paired_checkpoint_share_the_atomic_writer(
    artifact, tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(
        json_io, "write_json", lambda data, path: calls.append((data, path))
    )
    path = tmp_path / "artifact.json"
    artifacts.write_artifact(artifact, path)
    run_paired_performance._write_report({"verdict": "running"}, path)
    assert calls == [(artifact, path), ({"verdict": "running"}, path)]
