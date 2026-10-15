"""Validate fresh paired jobs, worker correctness and block uncertainty."""

import contextlib
import json
import os
from pathlib import Path

import pytest
from cgbench import cli

from scripts import run_paired_performance


def job():
    return {
        "case": {"name": "fixture"},
        "iterations": 4,
        "matvecs": 6,
        "input_sha256": "a" * 64,
        "residual_threshold": 1e-6,
    }


class FakeWorker:
    def __init__(self, name, elapsed, calls):
        self.name = name
        self.elapsed = elapsed
        self.calls = calls

    def request(self, operation, **arguments):
        assert operation == "sample"
        assert arguments["repetitions"] == 7
        self.calls.append(self.name)
        return {
            "elapsed_ns": self.elapsed,
            "iterations": 4,
            "matvecs": 6,
            "solver_status": "converged",
        }


def test_adjacent_rounds_reverse_worker_order_and_retain_raw_intervals():
    calls = []
    workers = {
        "baseline": FakeWorker("baseline", 700, calls),
        "current": FakeWorker("current", 714, calls),
    }
    rounds = [
        run_paired_performance._paired_round(workers, job(), index, 7)
        for index in range(2)
    ]
    assert calls == ["baseline", "current", "current", "baseline"]
    assert rounds[0]["order"] == ["baseline", "current"]
    assert rounds[1]["order"] == ["current", "baseline"]
    for row in rounds:
        assert row["baseline_elapsed_ns"] == 700
        assert row["current_elapsed_ns"] == 714


@pytest.mark.parametrize(
    "mismatch", ["hash", "counts", "convergence", "gate", "threshold"]
)
def test_numerical_or_input_mismatches_abort_before_accepting_samples(
    mismatch,
):
    expected = job()
    response = {
        **expected,
        "solver_status": "converged",
        "eligible": True,
    }
    if mismatch == "hash":
        response["input_sha256"] = "b" * 64
    elif mismatch == "counts":
        response["matvecs"] += 1
    elif mismatch == "convergence":
        response["solver_status"] = "max_iterations"
    elif mismatch == "gate":
        response["eligible"] = False
    else:
        response["residual_threshold"] *= 2
    with pytest.raises(ValueError):
        run_paired_performance._verify_work(
            expected, response, require_hash=True
        )


def test_contiguous_block_interval_preserves_sustained_variation():
    low, high = run_paired_performance._interval([1.02] * 50, resamples=1000)
    assert low == pytest.approx(1.02)
    assert high == pytest.approx(1.02)
    ratios = [0.98] * 25 + [1.08] * 25
    interval = run_paired_performance._interval(ratios, resamples=1000)
    assert interval == run_paired_performance._interval(ratios, resamples=1000)
    assert interval[0] < 1.05 < interval[1]
    with pytest.raises(ValueError, match="complete blocks"):
        run_paired_performance._interval(ratios[:-1], resamples=1000)


def paired_arguments(tmp_path, *options):
    return [
        "--baseline-pythonpath",
        "baseline",
        "--current-pythonpath",
        "current",
        "--output",
        str(tmp_path / "paired.json"),
        *options,
    ]


def test_fresh_jobs_use_shared_suites_and_solver_controls(tmp_path):
    options = [
        "--suite",
        "smoke-sparse-solvers",
        "--scope",
        "api",
        "--implementations",
        "native-cg,native-pcg",
        "--seed",
        "42",
        "--rtol",
        "1e-7",
        "--atol",
        "1e-11",
        "--maxiter",
        "800",
        "--residual-refresh",
        "23",
        "--no-fused",
    ]
    arguments = run_paired_performance.parser().parse_args(
        paired_arguments(tmp_path, *options)
    )
    shared = cli.benchmark_spec(cli.parser().parse_args(["run", *options]))
    resolved = cli.benchmark_spec(arguments)
    assert resolved == shared
    jobs = run_paired_performance._jobs(resolved)
    assert len(jobs) == 6
    assert {entry["implementation"] for entry in jobs} == {
        "native-cg",
        "native-pcg",
    }
    assert all(
        entry["seed"] == 42 and entry["scope"] == "api" for entry in jobs
    )
    assert all(entry["options"] == resolved["solver"] for entry in jobs)
    assert all(
        "iterations" not in entry and "input_sha256" not in entry
        for entry in jobs
    )


@pytest.mark.parametrize(
    "arguments,message",
    [
        (["--threads", "2"], "one native CPU thread"),
        (["--implementations", "numpy-out"], "native CPU candidates"),
        (["--rounds", "49"], "complete blocks"),
        (["--rounds", "45"], "complete blocks"),
        (["--rounds", "55"], "worker-order cycles"),
        (["--block-length", "0"], "block-length"),
        (["--bootstrap-samples", "99"], "bootstrap-samples"),
        (["--warmups", "0"], "warmups"),
        (["--max-repetitions", "0"], "max-repetitions"),
        (["--target-sample-ms", "nan"], "target-sample-ms"),
        (["--target-sample-ms", "0"], "target-sample-ms"),
    ],
)
def test_invalid_modes_abort_before_workers(
    arguments, message, tmp_path, monkeypatch, capsys
):
    def forbidden(*args):
        raise AssertionError("invalid arguments reached a worker")

    monkeypatch.setattr(run_paired_performance, "Worker", forbidden)
    with pytest.raises(SystemExit):
        run_paired_performance.main(paired_arguments(tmp_path, *arguments))
    assert message in capsys.readouterr().err


class ProtocolWorker:
    """Predictable timings keep launcher tests independent of machine speed."""

    def __init__(self, name):
        self.name = name

    def request(self, operation, **arguments):
        if operation in ("prepare", "verify"):
            return {
                **job(),
                "solver_status": "converged",
                "eligible": True,
                "true_residual_norm": 1e-9,
            }
        if operation == "sample":
            return {
                "iterations": 4,
                "matvecs": 6,
                "solver_status": "converged",
                "elapsed_ns": arguments["repetitions"]
                * (10000 if self.name == "baseline" else 10100),
            }
        assert operation == "metadata"
        return {"native": {"module_sha256": self.name}}

    def close(self):
        pass


def test_launcher_records_new_expected_work_without_comparison_files(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(run_paired_performance, "Worker", ProtocolWorker)
    assert (
        run_paired_performance.main(
            paired_arguments(
                tmp_path,
                "--suite",
                "smoke",
                "--seed",
                "17",
                "--rounds",
                "10",
                "--block-length",
                "1",
                "--bootstrap-samples",
                "100",
                "--warmups",
                "1",
                "--target-sample-ms",
                "0.001",
                "--max-repetitions",
                "1",
            )
        )
        == 0
    )
    result = json.loads((tmp_path / "paired.json").read_text())
    assert "comparison_files" not in result
    assert result["spec"]["seed"] == 17
    assert result["protocol"]["bootstrap_blocks"] == 10
    assert result["verdict"] == "supported_no_material_regression"
    (case,) = result["cases"]
    assert case["case"]["name"] == "dense-16"
    assert case["iterations"] == 4
    assert case["input_sha256"] == "a" * 64
    assert case["preparation"] == case["verification"]
    assert case["ratio"] == pytest.approx(1.01)
    assert len(case["rounds"]) == 10
    assert result["metadata"] == result["final_metadata"]


@pytest.mark.parametrize("mismatch", ["hash", "work", "gate", "threshold"])
def test_prepare_rejects_disagreement_between_builds(mismatch):
    class MismatchedWorker(ProtocolWorker):
        def request(self, operation, **arguments):
            result = super().request(operation, **arguments)
            if mismatch == "hash":
                result["input_sha256"] = "b" * 64
            elif mismatch == "work":
                result["iterations"] += 1
            elif mismatch == "gate":
                result["eligible"] = False
            else:
                result["residual_threshold"] *= 2
            return result

    with pytest.raises(ValueError):
        run_paired_performance._prepare_job(
            {
                "baseline": ProtocolWorker("baseline"),
                "current": MismatchedWorker("current"),
            },
            {"case": {"name": "fresh"}},
        )


def test_launcher_rejects_same_binary_even_in_different_processes(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        run_paired_performance, "Worker", lambda _: ProtocolWorker("same")
    )
    with pytest.raises(ValueError, match="same native binary"):
        run_paired_performance.main(
            paired_arguments(tmp_path, "--suite", "smoke")
        )


def test_launcher_rejects_binary_change_during_measurement(
    tmp_path, monkeypatch
):
    class ChangingWorker(ProtocolWorker):
        reads = 0

        def request(self, operation, **arguments):
            result = super().request(operation, **arguments)
            if operation == "metadata":
                self.reads += 1
                result["native"]["module_sha256"] += str(self.reads)
            return result

    monkeypatch.setattr(run_paired_performance, "Worker", ChangingWorker)
    with pytest.raises(ValueError, match="binary changed"):
        run_paired_performance.main(
            paired_arguments(
                tmp_path,
                "--suite",
                "smoke",
                "--rounds",
                "10",
                "--block-length",
                "1",
                "--bootstrap-samples",
                "100",
            )
        )


def test_real_workers_generate_and_validate_fresh_fixtures_without_old_json(
    tmp_path,
):
    import cgbench
    import cgpy

    package_paths = dict.fromkeys(
        [
            str(Path(cgbench.__file__).resolve().parents[1]),
            str(Path(cgpy.__file__).resolve().parents[1]),
        ]
    )
    pythonpath = os.pathsep.join(package_paths)
    arguments = run_paired_performance.parser().parse_args(
        paired_arguments(
            tmp_path,
            "--suite",
            "smoke",
            "--seed",
            "19",
        )
    )
    (fresh_job,) = run_paired_performance._jobs(cli.benchmark_spec(arguments))
    with contextlib.ExitStack() as cleanup:
        workers = {}
        for variant in ("baseline", "current"):
            workers[variant] = run_paired_performance.Worker(pythonpath)
            cleanup.callback(workers[variant].close)
        expected, preparation = run_paired_performance._prepare_job(
            workers, fresh_job
        )
        metadata = {
            name: worker.request("metadata")
            for name, worker in workers.items()
        }
        rows = [
            run_paired_performance._paired_round(workers, expected, index, 1)
            for index in range(10)
        ]
        assert preparation["baseline"] == preparation["current"]
        assert (
            expected["iterations"] > 0
            and expected["matvecs"] >= expected["iterations"]
        )
        for variant, worker in workers.items():
            verification = worker.request("verify")
            run_paired_performance._verify_work(
                expected, verification, require_hash=True
            )
            assert (
                verification["true_residual_norm"]
                <= verification["residual_threshold"]
            )
            final = worker.request("metadata")
            assert (
                final["native"]["module_sha256"]
                == metadata[variant]["native"]["module_sha256"]
            )
        interval = run_paired_performance._interval(
            [
                row["current_elapsed_ns"] / row["baseline_elapsed_ns"]
                for row in rows
            ],
            block_length=1,
            resamples=100,
        )
        assert 0 < interval[0] <= interval[1]
    assert not (tmp_path / "paired.json").exists()
