"""Compare correctness-gated cgbench artifacts with block-aware uncertainty.

Supply every artifact from the balanced baseline/current run sequence. Each
artifact is one timing block; rounds are normalized by its repetition count.
The percentile bootstrap resamples artifacts, then rounds within artifacts.
It does not infer ABBA pairing or execution order from filenames.
"""

import argparse
import dataclasses
import json
import math
import pathlib
import random
import statistics

from cgbench import json_io, validation


@dataclasses.dataclass(frozen=True)
class Measurement:
    """Validated numerical work and normalized rounds for one candidate."""

    case: dict
    input_sha256: str
    iterations: int
    matvecs: int
    samples: tuple[float, ...]


@dataclasses.dataclass(frozen=True)
class Artifact:
    path: pathlib.Path
    signature: dict
    measurements: dict[tuple[str, str], Measurement]


def _positive_integer(value, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _read_artifact(path: pathlib.Path) -> Artifact:
    data = json.loads(path.read_text(encoding="utf8"))
    if data.get("schema_version") != 1:
        raise ValueError(f"{path}: unsupported cgbench schema")
    spec = data["spec"]
    validation.validate_scope(spec.get("scope"), spec.get("device", "cpu"))
    # Sampling may differ between files; every round carries its own divisor.
    signature = {
        "metric": data["metric"],
        "spec": {
            key: value for key, value in spec.items() if key != "sampling"
        },
    }
    declared_cases = {case["name"]: case for case in spec["cases"]}
    measurements = {}
    case_names = set()
    for case in data["cases"]:
        name = case["case"]["name"]
        if name in case_names or case["case"] != declared_cases.get(name):
            raise ValueError(f"{path}: duplicate or undeclared case {name}")
        case_names.add(name)
        sampling = case["sampling"]
        rounds = _positive_integer(sampling["rounds"], "rounds")
        repetitions = _positive_integer(sampling["repetitions"], "repetitions")
        candidates = set()
        for result in case["results"]:
            candidate = result["name"]
            if candidate in candidates:
                raise ValueError(f"{path}: duplicate candidate {candidate}")
            candidates.add(candidate)
            if result["status"] != "measured":
                raise ValueError(f"{path}: {name}/{candidate} is not measured")
            try:
                validation.validate_converged_result(result)
            except ValueError as error:
                raise ValueError(
                    f"{path}: {name}/{candidate}: {error}"
                ) from error
            elapsed = result["round_elapsed_ns"]
            if len(elapsed) != rounds or any(
                type(value) is not int or value <= 0 for value in elapsed
            ):
                raise ValueError(f"{path}: {name} has invalid timing rounds")
            counts = result["iterations"], result["matvecs"]
            digest = case["input_sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)
            ):
                raise ValueError(f"{path}: {name} has an invalid input hash")
            measurements[name, candidate] = Measurement(
                case=case["case"],
                input_sha256=digest,
                iterations=counts[0],
                matvecs=counts[1],
                samples=tuple(value / repetitions for value in elapsed),
            )
        if candidates != set(spec["implementations"]):
            raise ValueError(f"{path}: {name} has missing candidates")
    if not measurements or case_names != set(declared_cases):
        raise ValueError(f"{path}: missing benchmark cases")
    return Artifact(path, signature, measurements)


def _verify_comparable(artifacts: list[Artifact]) -> None:
    reference = artifacts[0]
    for artifact in artifacts[1:]:
        if artifact.signature != reference.signature:
            raise ValueError(f"{artifact.path}: benchmark spec/options differ")
        if artifact.measurements.keys() != reference.measurements.keys():
            raise ValueError(f"{artifact.path}: cases/candidates differ")
        for key, expected in reference.measurements.items():
            actual = artifact.measurements[key]
            if (
                actual.case != expected.case
                or actual.input_sha256 != expected.input_sha256
            ):
                raise ValueError(f"{artifact.path}: {key} input/spec differs")
            if (actual.iterations, actual.matvecs) != (
                expected.iterations,
                expected.matvecs,
            ):
                raise ValueError(
                    f"{artifact.path}: {key} numerical work differs"
                )


def _resampled_median(
    blocks: list[tuple[float, ...]], rng: random.Random
) -> float:
    samples = []
    for _ in blocks:
        block = rng.choice(blocks)
        samples.extend(rng.choices(block, k=len(block)))
    return statistics.median(samples)


def _bootstrap_interval(
    baseline: list[tuple[float, ...]],
    current: list[tuple[float, ...]],
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        old = _resampled_median(baseline, rng)
        new = _resampled_median(current, rng)
        ratios.append(new / old)
    quantiles = statistics.quantiles(ratios, n=40, method="inclusive")
    return quantiles[0], quantiles[-1]


def _summary(blocks: list[tuple[float, ...]]) -> dict:
    samples = [value for block in blocks for value in block]
    return {
        "artifact_blocks": len(blocks),
        "rounds": len(samples),
        "median_ns_per_call": statistics.median(samples),
        "block_medians_ns_per_call": [
            statistics.median(block) for block in blocks
        ],
    }


def compare(
    baseline_paths: list[pathlib.Path],
    current_paths: list[pathlib.Path],
    *,
    max_regression: float = 0.05,
    resamples: int = 10000,
    seed: int = 20261015,
) -> dict:
    """Compare all supplied blocks, rejecting incompatible numerical work.

    Ratios are current/baseline: values above one indicate a slowdown.
    An upper confidence bound within budget supports the measured claim.
    A lower bound above budget signals regression; overlap is inconclusive.
    At least two artifact blocks per side are required to support a pass.
    """
    if not math.isfinite(max_regression) or max_regression < 0:
        raise ValueError("max-regression must be finite and nonnegative")
    if type(resamples) is not int or resamples < 100:
        raise ValueError("bootstrap-samples must be an integer >= 100")
    if not baseline_paths or not current_paths:
        raise ValueError("baseline and current each require artifact files")
    paths = [pathlib.Path(path).resolve() for path in baseline_paths]
    paths += [pathlib.Path(path).resolve() for path in current_paths]
    if len(set(paths)) != len(paths):
        raise ValueError(
            "artifact files must be distinct; do not reuse blocks"
        )
    baseline = [_read_artifact(path) for path in paths[: len(baseline_paths)]]
    current = [_read_artifact(path) for path in paths[len(baseline_paths) :]]
    _verify_comparable(baseline + current)
    limit = 1 + max_regression
    cases = []
    for index, key in enumerate(sorted(baseline[0].measurements)):
        old_blocks = [
            artifact.measurements[key].samples for artifact in baseline
        ]
        new_blocks = [
            artifact.measurements[key].samples for artifact in current
        ]
        old_summary, new_summary = _summary(old_blocks), _summary(new_blocks)
        ratio = (
            new_summary["median_ns_per_call"]
            / old_summary["median_ns_per_call"]
        )
        lower, upper = _bootstrap_interval(
            old_blocks, new_blocks, resamples, seed + index
        )
        if lower > limit:
            verdict = "regression"
        elif upper <= limit and min(len(baseline), len(current)) >= 2:
            verdict = "supported_no_material_regression"
        else:
            verdict = "inconclusive"
        reference = baseline[0].measurements[key]
        cases.append(
            {
                "case": reference.case,
                "input_sha256": reference.input_sha256,
                "implementation": key[1],
                "iterations": reference.iterations,
                "matvecs": reference.matvecs,
                "baseline": old_summary,
                "current": new_summary,
                "ratio": ratio,
                "measured_regression_fraction": ratio - 1,
                "ratio_ci95": {"lower": lower, "upper": upper},
                "verdict": verdict,
            }
        )
    verdicts = {case["verdict"] for case in cases}
    overall = (
        "regression"
        if "regression" in verdicts
        else "inconclusive"
        if "inconclusive" in verdicts
        else "supported_no_material_regression"
    )
    return {
        "schema_version": 1,
        "metric": "current_to_baseline_median_ns_per_call",
        "max_regression": max_regression,
        "ratio_limit": limit,
        "baseline_files": [str(artifact.path) for artifact in baseline],
        "current_files": [str(artifact.path) for artifact in current],
        "benchmark": baseline[0].signature,
        "bootstrap": {
            "confidence": 0.95,
            "resamples": resamples,
            "seed": seed,
            "method": "artifact blocks, then rounds; percentile median ratio",
        },
        "limitations": [
            "Intervals describe the supplied artifacts on the measured host.",
            "Few artifact blocks give limited evidence of process variance.",
            "Rounds are resampled independently within each artifact; serial "
            "dependence may make the interval too narrow.",
            "Run order, ABBA pairing, thermal state and background load are "
            "not recoverable from this artifact schema.",
            "Intervals are per case; simultaneous coverage is not adjusted.",
        ],
        "cases": cases,
        "verdict": overall,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=pathlib.Path, nargs="+", required=True
    )
    parser.add_argument(
        "--current", type=pathlib.Path, nargs="+", required=True
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--max-regression", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261015)
    options = parser.parse_args(argv)
    try:
        report = compare(
            options.baseline,
            options.current,
            max_regression=options.max_regression,
            resamples=options.bootstrap_samples,
            seed=options.seed,
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    json_io.write_json(report, options.output)
    for case in report["cases"]:
        interval = case["ratio_ci95"]
        print(
            f"{case['case']['name']}/{case['implementation']}: "
            f"ratio={case['ratio']:.4f}, "
            f"95%=[{interval['lower']:.4f}, {interval['upper']:.4f}], "
            f"{case['verdict']}"
        )
    print(f"Overall: {report['verdict']}; report={options.output.resolve()}")
    return int(report["verdict"] != "supported_no_material_regression")


if __name__ == "__main__":
    raise SystemExit(main())
