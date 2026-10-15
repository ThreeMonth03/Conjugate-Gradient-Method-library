"""Measure paired CPU runs in persistent baseline/current worker processes.

Use existing comparison reports to select identical fixtures and solver work.
Only execution inside each worker is timed; IPC is outside the interval.
"""

import argparse
import contextlib
import json
import math
import os
import pathlib
import random
import statistics
import subprocess
import sys
import time


class Worker:
    """A separate interpreter with explicit package paths and thread limits."""

    def __init__(self, pythonpath: str):
        environment = dict(os.environ)
        environment["PYTHONPATH"] = pythonpath
        environment.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
        for variable in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "BLIS_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[variable] = "1"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(pathlib.Path(__file__).resolve()),
                "--worker",
            ],
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

    def request(self, operation: str, **arguments) -> dict:
        self.process.stdin.write(
            json.dumps({"operation": operation, **arguments}) + "\n"
        )
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("performance worker exited unexpectedly")
        response = json.loads(line)
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def close(self) -> None:
        self.process.stdin.close()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.process.stdout.close()


def _worker() -> None:
    # Worker startup supplies thread limits before any numerical imports.
    from cgbench import artifacts, backends, cases, collector

    execute = None
    arrays = None
    options = None

    def checked_outcome() -> dict:
        outcome = execute()
        gate = collector.check_outcome(outcome, arrays, options)
        if not gate["eligible"]:
            raise ValueError(f"true-residual gate failed: {gate}")
        return {
            "iterations": outcome.iterations,
            "matvecs": outcome.matvecs,
            "solver_status": outcome.status,
            **gate,
        }

    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "prepare":
                job = request["job"]
                spec = cases.CaseSpec(**job["case"])
                fixture = cases.make_case(spec, job["seed"])
                arrays = fixture[:3]
                options = job["options"]
                execute = backends.make_executor(
                    job["implementation"], arrays, options, job["scope"], 1
                )
                response = {
                    "input_sha256": cases.input_hash(spec, fixture),
                    **checked_outcome(),
                }
            elif operation == "sample":
                start = time.perf_counter_ns()
                for _ in range(request["repetitions"]):
                    outcome = execute()
                elapsed = time.perf_counter_ns() - start
                response = {
                    "elapsed_ns": elapsed,
                    "iterations": outcome.iterations,
                    "matvecs": outcome.matvecs,
                    "solver_status": outcome.status,
                }
            elif operation == "verify":
                response = checked_outcome()
            elif operation == "metadata":
                response = artifacts.environment_metadata()
            else:
                raise ValueError(f"unknown worker operation: {operation}")
        except Exception as error:
            response = {"error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response, allow_nan=False), flush=True)


def _jobs(paths: list[pathlib.Path]) -> list[dict]:
    jobs = []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf8"))
        spec = report["benchmark"]["spec"]
        if spec.get("device", "cpu") != "cpu" or spec["threads"] != 1:
            raise ValueError("paired protocol requires one native CPU thread")
        for case in report["cases"]:
            if case["implementation"] not in (
                "native",
                "native-cg",
                "native-pcg",
            ):
                raise ValueError(
                    "paired protocol supports native CPU candidates"
                )
            jobs.append(
                {
                    "case": case["case"],
                    "suite": spec["suite"],
                    "scope": spec["scope"],
                    "implementation": case["implementation"],
                    "options": spec["solver"],
                    "seed": spec["seed"],
                    "input_sha256": case["input_sha256"],
                    "iterations": case["iterations"],
                    "matvecs": case["matvecs"],
                }
            )
    identities = {
        (
            job["suite"],
            job["scope"],
            job["case"]["name"],
            job["implementation"],
        )
        for job in jobs
    }
    if not jobs or len(identities) != len(jobs):
        raise ValueError(
            "comparison reports contain no jobs or duplicate jobs"
        )
    return jobs


def _verify_work(job: dict, response: dict, *, require_hash=False) -> None:
    keys = ("iterations", "matvecs")
    if require_hash:
        keys += ("input_sha256",)
    if any(response[key] != job[key] for key in keys):
        raise ValueError(
            f"{job['case']['name']}: input or numerical work differs"
        )
    if response["solver_status"] != "converged":
        raise ValueError(f"{job['case']['name']}: solve did not converge")
    if "eligible" in response and not response["eligible"]:
        raise ValueError(f"{job['case']['name']}: true-residual gate failed")


def _paired_round(workers: dict, job: dict, index: int, repetitions: int):
    order = ["baseline", "current"]
    if index % 2:
        order.reverse()
    result = {"order": order}
    for variant in order:
        response = workers[variant].request("sample", repetitions=repetitions)
        _verify_work(job, response)
        if response["elapsed_ns"] <= 0:
            raise ValueError("worker returned a nonpositive timing interval")
        result[variant + "_elapsed_ns"] = response["elapsed_ns"]
    return result


def _interval(
    ratios: list[float], *, block_length=5, resamples=10000, seed=20261015
) -> tuple[float, float]:
    """Percentile bootstrap over contiguous, nonoverlapping paired blocks."""
    if len(ratios) % block_length or len(ratios) < 10 * block_length:
        raise ValueError("paired bootstrap requires >= 10 complete blocks")
    blocks = [
        ratios[index : index + block_length]
        for index in range(0, len(ratios), block_length)
    ]
    rng = random.Random(seed)
    medians = []
    for _ in range(resamples):
        selected = rng.choices(blocks, k=len(blocks))
        medians.append(statistics.median(sum(selected, [])))
    quantiles = statistics.quantiles(medians, n=40, method="inclusive")
    return quantiles[0], quantiles[-1]


def _write_report(report: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparisons", nargs="+", type=pathlib.Path, required=True
    )
    parser.add_argument("--baseline-pythonpath", required=True)
    parser.add_argument("--current-pythonpath", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-regression", type=float, default=0.05)
    options = parser.parse_args(argv)
    if not math.isfinite(options.max_regression) or options.max_regression < 0:
        parser.error("max-regression must be finite and nonnegative")
    jobs = _jobs(options.comparisons)
    report = {
        "schema_version": 1,
        "metric": "median_paired_current_to_baseline_ns_per_call",
        "protocol": {
            "rounds": 50,
            "warmups": 5,
            "target_sample_ns": 10000000,
            "max_repetitions": 1000,
            "order": "alternating baseline/current, current/baseline",
            "timing": "worker-local clock; includes adapter output reset",
            "workers": "two persistent interpreters; one native CPU thread",
            "bootstrap": "10000 resamples of 10 contiguous paired blocks of 5",
            "bootstrap_seed": 20261015,
        },
        "max_regression": options.max_regression,
        "comparison_files": [
            str(path.resolve()) for path in options.comparisons
        ],
        "pythonpaths": {
            "baseline": options.baseline_pythonpath,
            "current": options.current_pythonpath,
        },
        "limitations": [
            "This follow-up protocol retains earlier artifact comparisons.",
            "Intervals are conditional on these two persistent worker processes.",
            "Alternating adjacent pairs limits drift but does not remove all "
            "thermal, scheduler, cache or background-load effects.",
            "Block length 5 covers short serial dependence; longer dependence "
            "or another host may yield different uncertainty.",
            "Intervals are per case, without simultaneous coverage adjustment.",
        ],
        "cases": [],
        "verdict": "running",
    }
    with contextlib.ExitStack() as cleanup:
        workers = {}
        for variant, pythonpath in report["pythonpaths"].items():
            workers[variant] = Worker(pythonpath)
            cleanup.callback(workers[variant].close)
        for index, job in enumerate(jobs):
            preparation = {}
            for variant, worker in workers.items():
                preparation[variant] = worker.request("prepare", job=job)
                _verify_work(job, preparation[variant], require_hash=True)
            if not index:
                report["metadata"] = {
                    variant: worker.request("metadata")
                    for variant, worker in workers.items()
                }
                hashes = {
                    data["native"]["module_sha256"]
                    for data in report["metadata"].values()
                }
                if len(hashes) != 2:
                    raise ValueError("workers loaded the same native binary")
            warmups = [
                _paired_round(workers, job, count, 1) for count in range(5)
            ]
            fastest = min(
                statistics.median(
                    row[variant + "_elapsed_ns"] for row in warmups
                )
                for variant in workers
            )
            repetitions = min(1000, max(1, math.ceil(10000000 / fastest)))
            rounds = [
                _paired_round(workers, job, count, repetitions)
                for count in range(50)
            ]
            verification = {}
            for variant, worker in workers.items():
                verification[variant] = worker.request("verify")
                _verify_work(job, verification[variant])
            ratios = [
                row["current_elapsed_ns"] / row["baseline_elapsed_ns"]
                for row in rounds
            ]
            lower, upper = _interval(ratios, seed=20261015 + index)
            limit = 1 + options.max_regression
            verdict = (
                "regression"
                if lower > limit
                else "supported_no_material_regression"
                if upper <= limit
                else "inconclusive"
            )
            report["cases"].append(
                {
                    **job,
                    "repetitions": repetitions,
                    "warmups": warmups,
                    "rounds": rounds,
                    "baseline_median_ns_per_call": statistics.median(
                        row["baseline_elapsed_ns"] / repetitions
                        for row in rounds
                    ),
                    "current_median_ns_per_call": statistics.median(
                        row["current_elapsed_ns"] / repetitions
                        for row in rounds
                    ),
                    "pair_ratios": ratios,
                    "ratio": statistics.median(ratios),
                    "ratio_ci95": {"lower": lower, "upper": upper},
                    "preparation": preparation,
                    "verification": verification,
                    "verdict": verdict,
                }
            )
            _write_report(report, options.output)
            print(
                f"{job['suite']}/{job['scope']}/{job['case']['name']}/"
                f"{job['implementation']}: median={statistics.median(ratios):.4f}, "
                f"95%=[{lower:.4f}, {upper:.4f}], {verdict}",
                flush=True,
            )
        report["final_metadata"] = {
            variant: worker.request("metadata")
            for variant, worker in workers.items()
        }
        for variant in workers:
            before = report["metadata"][variant]["native"]["module_sha256"]
            after = report["final_metadata"][variant]["native"][
                "module_sha256"
            ]
            if before != after:
                raise ValueError(
                    "native binary changed during paired measurement"
                )
    verdicts = {case["verdict"] for case in report["cases"]}
    report["verdict"] = (
        "regression"
        if "regression" in verdicts
        else "inconclusive"
        if "inconclusive" in verdicts
        else "supported_no_material_regression"
    )
    _write_report(report, options.output)
    return int(report["verdict"] != "supported_no_material_regression")


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        _worker()
    else:
        raise SystemExit(main())
