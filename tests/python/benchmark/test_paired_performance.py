"""Validate paired scheduling and serial-block uncertainty without timing work."""

import pytest

from scripts import run_paired_performance


def job():
    return {
        "case": {"name": "fixture"},
        "iterations": 4,
        "matvecs": 6,
        "input_sha256": "a" * 64,
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


@pytest.mark.parametrize("mismatch", ["hash", "counts", "convergence", "gate"])
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
    else:
        response["eligible"] = False
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
