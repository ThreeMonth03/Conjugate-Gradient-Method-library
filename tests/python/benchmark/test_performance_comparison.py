"""Validate timing units, numerical comparability and uncertainty decisions."""

import copy
import json

import pytest

from scripts import compare_performance


def artifact(elapsed, *, repetitions=1):
    case = {"name": "dense-4", "size": 4, "condition_scale": 100.0}
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
            "sampling": {
                "rounds": len(elapsed),
                "repetitions": repetitions,
                "warmups": 2,
            },
        },
        "cases": [
            {
                "case": case,
                "input_sha256": "a" * 64,
                "sampling": {
                    "rounds": len(elapsed),
                    "repetitions": repetitions,
                    "warmups": 2,
                },
                "results": [
                    {
                        "name": "native",
                        "status": "measured",
                        "solver_status": "converged",
                        "iterations": 4,
                        "matvecs": 6,
                        "true_residual_norm": 1e-10,
                        "residual_threshold": 1e-8,
                        "round_elapsed_ns": elapsed,
                    }
                ],
            }
        ],
    }


def save_blocks(tmp_path, prefix, artifacts):
    paths = []
    for index, data in enumerate(artifacts):
        path = tmp_path / f"{prefix}-{index}.json"
        path.write_text(json.dumps(data), encoding="utf8")
        paths.append(path)
    return paths


@pytest.mark.parametrize(
    "elapsed,verdict",
    [
        (102, "supported_no_material_regression"),
        (110, "regression"),
    ],
)
def test_constant_timings_respect_the_five_percent_budget(
    tmp_path, elapsed, verdict
):
    baseline = save_blocks(tmp_path, "old", [artifact([100] * 4)] * 2)
    current = save_blocks(tmp_path, "new", [artifact([elapsed] * 4)] * 2)
    report = compare_performance.compare(baseline, current, resamples=100)
    assert report["verdict"] == verdict
    row = report["cases"][0]
    assert row["ratio"] == pytest.approx(elapsed / 100)
    assert row["ratio_ci95"]["lower"] == pytest.approx(elapsed / 100)
    assert row["ratio_ci95"]["upper"] == pytest.approx(elapsed / 100)


def test_rounds_are_normalized_before_pooling_and_results_are_deterministic(
    tmp_path,
):
    baseline = save_blocks(
        tmp_path,
        "old",
        [
            artifact([100] * 3, repetitions=10),
            artifact([20] * 5, repetitions=2),
        ],
    )
    current = save_blocks(
        tmp_path, "new", [artifact([30] * 4, repetitions=3)] * 2
    )
    first = compare_performance.compare(baseline, current, resamples=100)
    second = compare_performance.compare(baseline, current, resamples=100)
    assert first == second
    row = first["cases"][0]
    assert row["baseline"]["median_ns_per_call"] == 10
    assert row["current"]["median_ns_per_call"] == 10
    assert row["baseline"]["rounds"] == 8
    assert row["ratio"] == 1


@pytest.mark.parametrize("mismatch", ["options", "hash", "work", "status"])
def test_incompatible_or_unvalidated_work_is_rejected(tmp_path, mismatch):
    old = artifact([100] * 4)
    new = copy.deepcopy(old)
    if mismatch == "options":
        new["spec"]["scope"] = "api"
    elif mismatch == "hash":
        new["cases"][0]["input_sha256"] = "b" * 64
    elif mismatch == "work":
        new["cases"][0]["results"][0]["matvecs"] += 1
    else:
        new["cases"][0]["results"][0]["status"] = "invalid"
    baseline = save_blocks(tmp_path, "old", [old])
    current = save_blocks(tmp_path, "new", [new])
    with pytest.raises(ValueError):
        compare_performance.compare(baseline, current, resamples=100)


def test_block_variation_and_a_single_run_cannot_support_a_pass(tmp_path):
    baseline = save_blocks(tmp_path, "old", [artifact([100] * 20)] * 2)
    current = save_blocks(
        tmp_path, "new", [artifact([99] * 20), artifact([108] * 20)]
    )
    report = compare_performance.compare(baseline, current, resamples=1000)
    row = report["cases"][0]
    assert row["ratio"] < 1.05
    assert row["ratio_ci95"]["lower"] < 1.05
    assert row["ratio_ci95"]["upper"] > 1.05
    assert report["verdict"] == "inconclusive"
    single = compare_performance.compare(
        baseline[:1], current[:1], resamples=100
    )
    assert single["verdict"] == "inconclusive"


def test_cli_saves_the_report_and_fails_on_inconclusive_evidence(tmp_path):
    baseline = save_blocks(tmp_path, "old", [artifact([100] * 4)])
    current = save_blocks(tmp_path, "new", [artifact([100] * 4)])
    output = tmp_path / "comparison.json"
    code = compare_performance.main(
        [
            "--baseline",
            str(baseline[0]),
            "--current",
            str(current[0]),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
        ]
    )
    assert code == 1
    assert json.loads(output.read_text())["verdict"] == "inconclusive"
