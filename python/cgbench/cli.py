"""Launch comparisons after setting thread limits in a fresh process."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

IMPLEMENTATIONS = (
    "native",
    "numpy-out",
    "scipy-out",
    "torch-eager-out",
    "native-cg",
    "native-pcg",
    "scipy-cg",
    "scipy-pcg",
    "torch-cg",
    "torch-pcg",
    "scipy-reference-cg",
    "scipy-reference-pcg",
    "scipy-lu-total",
    "scipy-lu-reuse",
)
SPARSE_SOLVERS = (
    "native-cg",
    "native-pcg",
    "scipy-cg",
    "scipy-pcg",
    "scipy-lu-total",
    "scipy-lu-reuse",
)


def _positive(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def _nonnegative(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def parser():
    from .cases import SUITES

    result = argparse.ArgumentParser(
        description="Reproducible CPU CG comparisons (also run by make pyprof)"
    )
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--suite", choices=tuple(SUITES), default="cpu")
    run.add_argument(
        "--implementations", help="comma-separated candidates; default follows suite storage"
    )
    run.add_argument("--scope", choices=("api", "prepared"), default="prepared")
    run.add_argument("--threads", type=_positive, default=1)
    run.add_argument("--seed", type=_nonnegative, default=20261015)
    run.add_argument("--warmups", type=_nonnegative, default=3)
    run.add_argument("--repetitions", type=_positive, default=3)
    run.add_argument("--rounds", type=_positive, default=30)
    run.add_argument("--algorithm", choices=("standard", "explicit"), default="standard")
    run.add_argument("--rtol", type=float, default=1e-8)
    run.add_argument("--atol", type=float, default=1e-12)
    run.add_argument("--maxiter", type=_nonnegative, default=2000)
    run.add_argument("--residual-refresh", type=_nonnegative, default=50)
    run.add_argument("--preconditioner", choices=("none", "jacobi"), default="none")
    run.add_argument("--output", default="build/benchmarks/cpu.json")
    compare = commands.add_parser("compare")
    compare.add_argument("artifacts", nargs="+")
    replay = commands.add_parser("replay")
    replay.add_argument("artifact")
    replay.add_argument("--output", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--spec-json", required=True)
    worker.add_argument("--output", required=True)
    return result


def _launch(spec, output):
    from .artifacts import THREAD_VARIABLES

    env = dict(os.environ)
    env.update({name: str(spec["threads"]) for name in THREAD_VARIABLES})
    env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    command = [
        sys.executable,
        "-m",
        "cgbench",
        "_worker",
        "--spec-json",
        json.dumps(spec),
        "--output",
        os.fspath(output),
    ]
    return subprocess.run(command, env=env, check=False).returncode


def _run_worker(spec, output):
    # Imports happen only after the launcher sets the child environment.
    from .artifacts import environment_metadata, print_summary, write_artifact
    from .cases import CaseSpec
    from .collector import collect_case

    cases = [collect_case(CaseSpec(**entry), spec) for entry in spec["cases"]]
    data = {
        "schema_version": 1,
        "metric": "sparse_solver_comparison"
        if spec.get("comparison") == "sparse-solvers"
        else "time_to_solution",
        "spec": spec,
        "environment": environment_metadata(),
        "cases": cases,
    }
    write_artifact(data, output)
    print_summary(data)
    print(f"Saved {Path(output).resolve()}")
    failed = any(
        entry["status"] == "invalid"
        or (entry["name"].startswith("native") and entry["status"] == "ineligible")
        for case in cases
        for entry in case["results"]
    )
    if spec.get("expected_hashes"):
        hashes = {case["case"]["name"]: case["input_sha256"] for case in cases}
        if hashes != spec["expected_hashes"]:
            print(
                "Replay input hash mismatch; results are not comparable.",
                file=sys.stderr,
            )
            failed = True
    return int(failed)


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "_worker":
        return _run_worker(json.loads(arguments.spec_json), arguments.output)
    if arguments.command == "run":
        import math

        from .cases import SUITES

        default_reference = (
            "scipy-out" if SUITES[arguments.suite][0].storage == "csr" else "numpy-out"
        )
        comparison = (
            "sparse-solvers"
            if arguments.suite in ("sparse-solvers", "smoke-sparse-solvers", "sparse-solvers-large")
            else "implementations"
        )
        names = (
            arguments.implementations.split(",")
            if arguments.implementations
            else list(SPARSE_SOLVERS)
            if comparison == "sparse-solvers"
            else ["native", default_reference, "torch-eager-out"]
        )
        if (
            not names
            or len(names) != len(set(names))
            or any(name not in IMPLEMENTATIONS for name in names)
        ):
            raise SystemExit("implementations must be unique names: " + ",".join(IMPLEMENTATIONS))
        if any(
            name in ("scipy-cg", "scipy-pcg", "scipy-lu-total", "scipy-lu-reuse") for name in names
        ):
            comparison = "sparse-solvers"
        if not all(
            math.isfinite(value) and value >= 0 for value in (arguments.rtol, arguments.atol)
        ):
            raise SystemExit("rtol and atol must be finite and nonnegative")
        if arguments.algorithm != "standard" and (
            arguments.preconditioner == "jacobi" or any(name.endswith("pcg") for name in names)
        ):
            raise SystemExit("Jacobi PCG requires --algorithm standard")
        spec = {
            "comparison": comparison,
            "suite": arguments.suite,
            "cases": [case.to_dict() for case in SUITES[arguments.suite]],
            "implementations": names,
            "scope": arguments.scope,
            "threads": arguments.threads,
            "seed": arguments.seed,
            "sampling": {
                "warmups": arguments.warmups,
                "repetitions": arguments.repetitions,
                "rounds": arguments.rounds,
            },
            "solver": {
                "algorithm": arguments.algorithm,
                "rtol": arguments.rtol,
                "atol": arguments.atol,
                "maxiter": arguments.maxiter,
                "residual_refresh": arguments.residual_refresh,
                "fused": True,
                "preconditioner": arguments.preconditioner,
            },
        }
        return _launch(spec, arguments.output)
    from .artifacts import load_artifact, print_summary

    if arguments.command == "compare":
        artifacts = [load_artifact(path) for path in arguments.artifacts]
        first = artifacts[0]
        for path, artifact in zip(arguments.artifacts, artifacts, strict=False):
            print(f"Artifact: {path}")
            if artifact["spec"] != first["spec"] or artifact["environment"] != first["environment"]:
                print("Specification or environment differs; inspect metadata before comparing.")
            print_summary(artifact)
        return 0
    artifact = load_artifact(arguments.artifact)
    spec = dict(artifact["spec"])
    spec["expected_hashes"] = {
        case["case"]["name"]: case["input_sha256"] for case in artifact["cases"]
    }
    spec["expected_candidates"] = {
        case["case"]["name"]: [
            entry["name"] for entry in case["results"] if entry["status"] == "measured"
        ]
        for case in artifact["cases"]
    }
    spec["resolved_sampling"] = {
        case["case"]["name"]: case["sampling"] for case in artifact["cases"]
    }
    # Preserve each saved cycle count (the worker resolves the same candidate
    # set); unavailable candidates and versions remain visible in metadata.
    return _launch(spec, arguments.output)
