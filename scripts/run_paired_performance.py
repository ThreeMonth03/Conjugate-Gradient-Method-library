"""Measure paired CPU runs with fresh baseline/current workers for each case.

Generate fixtures from cgbench suites and derive numerical work from both builds.
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
    from cgbench import artifacts, cases, collector, execution

    execute = None
    arrays = None
    options = None
    spec = None
    fixture = None

    def checked_outcome() -> dict:
        outcome = execute()
        gate = collector.check_outcome(outcome, arrays, options)
        if not gate["eligible"]:
            raise ValueError(f"true-residual gate failed: {gate}")
        return {
            "input_sha256": cases.input_hash(spec, fixture),
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
                execute = execution.make_executor(
                    job["implementation"], arrays, options, job["scope"], 1
                )
                response = checked_outcome()
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


def _jobs(spec: dict) -> list[dict]:
    """Resolve fresh measurement work from the shared suite specification."""
    if spec.get("device", "cpu") != "cpu" or spec["threads"] != 1:
        raise ValueError("paired protocol requires one native CPU thread")
    if any(
        name not in ("native", "native-cg", "native-pcg")
        for name in spec["implementations"]
    ):
        raise ValueError("paired protocol supports native CPU candidates")
    return [
        {
            "case": case,
            "suite": spec["suite"],
            "scope": spec["scope"],
            "implementation": name,
            "options": spec["solver"],
            "seed": spec["seed"],
        }
        for case in spec["cases"]
        for name in spec["implementations"]
    ]


def _prepare_job(workers: dict, job: dict) -> tuple[dict, dict]:
    """Require both builds to agree before deriving this run's expected work."""
    preparation = {
        variant: worker.request("prepare", job=job)
        for variant, worker in workers.items()
    }
    reference = preparation["baseline"]
    expected = {
        **job,
        **{
            key: reference[key]
            for key in (
                "input_sha256",
                "iterations",
                "matvecs",
                "residual_threshold",
            )
        },
    }
    for response in preparation.values():
        _verify_work(expected, response, require_hash=True)
    return expected, preparation


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
    if "eligible" in response:
        from cgbench import validation

        if not response["eligible"]:
            raise ValueError(
                f"{job['case']['name']}: true-residual gate failed"
            )
        validation.validate_converged_result(response)
        if response["residual_threshold"] != job["residual_threshold"]:
            raise ValueError(
                f"{job['case']['name']}: residual threshold differs"
            )


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


def _validate_sampling(rounds: int, block_length: int, resamples: int) -> None:
    if type(block_length) is not int or block_length < 1:
        raise ValueError("block-length must be a positive integer")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("bootstrap-samples must be an integer >= 100")
    if (
        type(rounds) is not int
        or rounds % block_length
        or rounds < 10 * block_length
    ):
        raise ValueError("paired bootstrap requires >= 10 complete blocks")
    if rounds % 2:
        raise ValueError(
            "rounds must cover complete alternating worker-order cycles"
        )


def _interval(
    ratios: list[float], *, block_length=5, resamples=10000, seed=20261015
) -> tuple[float, float]:
    """Percentile bootstrap over contiguous, nonoverlapping paired blocks."""
    _validate_sampling(len(ratios), block_length, resamples)
    if any(not math.isfinite(ratio) or ratio <= 0 for ratio in ratios):
        raise ValueError("paired timing ratios must be finite and positive")
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
    from cgbench import json_io

    json_io.write_json(report, path)


def parser() -> argparse.ArgumentParser:
    from cgbench.cli import configure_benchmark_arguments

    result = argparse.ArgumentParser(description=__doc__)
    configure_benchmark_arguments(result, default_implementations="native")
    result.add_argument("--baseline-pythonpath", required=True)
    result.add_argument("--current-pythonpath", required=True)
    result.add_argument("--output", type=pathlib.Path, required=True)
    result.add_argument("--max-regression", type=float, default=0.05)
    result.add_argument("--warmups", type=int, default=5)
    result.add_argument("--rounds", type=int, default=50)
    result.add_argument("--target-sample-ms", type=float, default=10.0)
    result.add_argument("--max-repetitions", type=int, default=1000)
    result.add_argument("--block-length", type=int, default=5)
    result.add_argument("--bootstrap-samples", type=int, default=10000)
    return result


def main(argv: list[str] | None = None) -> int:
    from cgbench.cli import benchmark_spec

    argument_parser = parser()
    options = argument_parser.parse_args(argv)
    if not math.isfinite(options.max_regression) or options.max_regression < 0:
        argument_parser.error("max-regression must be finite and nonnegative")
    if options.warmups < 1 or options.max_repetitions < 1:
        argument_parser.error("warmups and max-repetitions must be positive")
    if (
        not math.isfinite(options.target_sample_ms)
        or options.target_sample_ms <= 0
    ):
        argument_parser.error("target-sample-ms must be finite and positive")
    try:
        _validate_sampling(
            options.rounds, options.block_length, options.bootstrap_samples
        )
        spec = benchmark_spec(options)
        jobs = _jobs(spec)
    except ValueError as error:
        argument_parser.error(str(error))
    target_sample_ns = math.ceil(options.target_sample_ms * 1000000)
    report = {
        "schema_version": 1,
        "metric": "median_paired_current_to_baseline_ns_per_call",
        "protocol": {
            "rounds": options.rounds,
            "warmups": options.warmups,
            "target_sample_ns": target_sample_ns,
            "max_repetitions": options.max_repetitions,
            "order": "alternating baseline/current, current/baseline",
            "timing": "worker-local clock; includes adapter output reset",
            "workers": "two fresh interpreters per case; one native CPU thread",
            "metadata": "per case, after preparation and final numerical verification",
            "bootstrap": "percentile resampling of contiguous paired blocks",
            "bootstrap_samples": options.bootstrap_samples,
            "bootstrap_blocks": options.rounds // options.block_length,
            "block_length": options.block_length,
            "bootstrap_seed": options.seed,
        },
        "max_regression": options.max_regression,
        "spec": spec,
        "pythonpaths": {
            "baseline": options.baseline_pythonpath,
            "current": options.current_pythonpath,
        },
        "limitations": [
            "Intervals are conditional on each case's two worker processes.",
            "Alternating adjacent pairs limits drift but does not remove all "
            "thermal, scheduler, cache or background-load effects.",
            "Finite blocks cover only short serial dependence; longer "
            "dependence or another host may yield different uncertainty.",
            "Intervals are per case, without simultaneous coverage adjustment.",
        ],
        "cases": [],
        "verdict": "running",
    }
    for index, job in enumerate(jobs):
        with contextlib.ExitStack() as cleanup:
            workers = {}
            for variant, pythonpath in report["pythonpaths"].items():
                workers[variant] = Worker(pythonpath)
                cleanup.callback(workers[variant].close)
            job, preparation = _prepare_job(workers, job)
            metadata = {
                variant: worker.request("metadata")
                for variant, worker in workers.items()
            }
            if not index:
                report["metadata"] = metadata
            for variant in workers:
                if (
                    metadata[variant]["native"]["module_sha256"]
                    != report["metadata"][variant]["native"]["module_sha256"]
                ):
                    raise ValueError(
                        "native binary changed between paired cases"
                    )
            hashes = {
                data["native"]["module_sha256"] for data in metadata.values()
            }
            if len(hashes) != 2:
                raise ValueError("workers loaded the same native binary")
            warmups = [
                _paired_round(workers, job, count, 1)
                for count in range(options.warmups)
            ]
            fastest = min(
                statistics.median(
                    row[variant + "_elapsed_ns"] for row in warmups
                )
                for variant in workers
            )
            repetitions = min(
                options.max_repetitions,
                max(1, math.ceil(target_sample_ns / fastest)),
            )
            rounds = [
                _paired_round(workers, job, count, repetitions)
                for count in range(options.rounds)
            ]
            verification = {}
            for variant, worker in workers.items():
                verification[variant] = worker.request("verify")
                _verify_work(job, verification[variant], require_hash=True)
            final_metadata = {
                variant: worker.request("metadata")
                for variant, worker in workers.items()
            }
            for variant in workers:
                before = metadata[variant]["native"]["module_sha256"]
                after = final_metadata[variant]["native"]["module_sha256"]
                if before != after:
                    raise ValueError(
                        "native binary changed during paired measurement"
                    )
            report["final_metadata"] = final_metadata
            ratios = [
                row["current_elapsed_ns"] / row["baseline_elapsed_ns"]
                for row in rounds
            ]
            lower, upper = _interval(
                ratios,
                block_length=options.block_length,
                resamples=options.bootstrap_samples,
                seed=options.seed + index,
            )
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
                    "metadata": metadata,
                    "final_metadata": final_metadata,
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
