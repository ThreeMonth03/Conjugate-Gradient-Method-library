"""Launch comparisons after setting thread limits in a fresh process."""

import argparse
import json
import os
import pathlib
import subprocess
import sys

from . import backends

IMPLEMENTATIONS = (
    "native",
    "native-blas",
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
) + backends.CUDA_IMPLEMENTATIONS
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


def configure_benchmark_arguments(result, *, default_implementations=None):
    """Share fixture, execution-scope and solver controls across launchers."""
    from . import cases as case_module

    result.add_argument(
        "--suite", choices=tuple(case_module.SUITES), default="cpu"
    )
    result.add_argument(
        "--implementations",
        default=default_implementations,
        help="comma-separated candidates; "
        + (
            f"default: {default_implementations}"
            if default_implementations
            else "default follows suite storage"
        ),
    )
    result.add_argument(
        "--scope", choices=("api", "prepared"), default="prepared"
    )
    result.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    result.add_argument("--cuda-device", type=_nonnegative, default=0)
    result.add_argument("--threads", type=_positive, default=1)
    result.add_argument("--seed", type=_nonnegative, default=20261015)
    result.add_argument(
        "--algorithm", choices=("standard", "explicit"), default="standard"
    )
    result.add_argument(
        "--fused",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="select native fused kernels or their separate-operation reference path",
    )
    result.add_argument("--rtol", type=float, default=1e-8)
    result.add_argument("--atol", type=float, default=1e-12)
    result.add_argument("--maxiter", type=_nonnegative, default=2000)
    result.add_argument("--residual-refresh", type=_nonnegative, default=50)
    result.add_argument(
        "--preconditioner", choices=("none", "jacobi"), default="none"
    )


def parser():
    result = argparse.ArgumentParser(
        description="Reproducible explicit CPU/CUDA CG comparisons (also run by make pyprof)"
    )
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    configure_benchmark_arguments(run)
    run.add_argument("--warmups", type=_nonnegative, default=3)
    run.add_argument("--repetitions", type=_positive, default=3)
    run.add_argument("--rounds", type=_positive, default=30)
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
    from . import artifacts as artifact_module

    env = dict(os.environ)
    env.update(
        {
            name: str(spec["threads"])
            for name in artifact_module.THREAD_VARIABLES
        }
    )
    env.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    if spec.get("device", "cpu") == "cuda":
        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
    from . import artifacts as artifact_module
    from . import cases as case_module
    from . import collector

    if spec.get("device", "cpu") == "cuda":
        from . import cuda_backends

        cuda_backends.preflight_cuda(
            spec["implementations"], spec.get("cuda_device", 0)
        )
    cases = [
        collector.collect_case(case_module.CaseSpec(**entry), spec)
        for entry in spec["cases"]
    ]
    data = {
        "schema_version": 1,
        "metric": "sparse_solver_comparison"
        if spec.get("comparison") == "sparse-solvers"
        else "time_to_solution",
        "spec": {
            key: value
            for key, value in spec.items()
            if key
            not in (
                "expected_hashes",
                "expected_candidates",
                "resolved_sampling",
                "expected_round_orders",
            )
        },
        "environment": artifact_module.environment_metadata(
            device=spec.get("device", "cpu"),
            cuda_device=spec.get("cuda_device", 0),
        ),
        "cases": cases,
    }
    if spec.get("expected_hashes"):
        data["replay_validation"] = (
            "matched input hashes, eligible candidates, resolved sampling and actual timing schedule"
        )
    artifact_module.write_artifact(data, output)
    artifact_module.print_summary(data)
    print(f"Saved {pathlib.Path(output).resolve()}")
    failed = any(
        entry["status"] == "invalid"
        or (
            (
                entry["name"].startswith("native")
                or spec.get("device", "cpu") == "cuda"
            )
            and entry["status"] == "ineligible"
        )
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


def benchmark_spec(arguments):
    """Validate and resolve a benchmark request without starting workers."""
    import math

    from . import cases as case_module

    default_reference = (
        "scipy-out"
        if case_module.SUITES[arguments.suite][0].storage == "csr"
        else "numpy-out"
    )
    comparison = (
        "sparse-solvers"
        if arguments.suite
        in (
            "sparse-solvers",
            "smoke-sparse-solvers",
            "sparse-solvers-large",
        )
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
        raise SystemExit(
            "implementations must be unique names: "
            + ",".join(IMPLEMENTATIONS)
        )
    if arguments.device == "cuda":
        if not arguments.implementations:
            raise SystemExit(
                "--device cuda requires explicit --implementations"
            )
        if any(name not in backends.CUDA_IMPLEMENTATIONS for name in names):
            raise SystemExit(
                "--device cuda requires CUDA candidates; CPU fallback is disabled"
            )
        comparison = "implementations"
    elif any(
        name in backends.CUDA_IMPLEMENTATIONS for name in names
    ) or arguments.suite.startswith(("cuda-", "smoke-cuda-")):
        raise SystemExit(
            "CUDA candidates and suite aliases require --device cuda"
        )
    if "cuda-cublas" in names and any(
        case.storage != "dense" for case in case_module.SUITES[arguments.suite]
    ):
        raise SystemExit("cuda-cublas requires a dense suite")
    if "native-blas" in names:
        if any(
            case.storage != "dense"
            for case in case_module.SUITES[arguments.suite]
        ):
            raise SystemExit("native-blas requires a dense suite")
        if arguments.threads != 1:
            raise SystemExit("native-blas currently requires --threads 1")
    if any(
        name in ("scipy-cg", "scipy-pcg", "scipy-lu-total", "scipy-lu-reuse")
        for name in names
    ):
        comparison = "sparse-solvers"
    if not all(
        math.isfinite(value) and value >= 0
        for value in (arguments.rtol, arguments.atol)
    ):
        raise SystemExit("rtol and atol must be finite and nonnegative")
    if arguments.algorithm != "standard" and (
        arguments.preconditioner == "jacobi"
        or any(name.endswith("pcg") for name in names)
    ):
        raise SystemExit("Jacobi PCG requires --algorithm standard")
    spec = {
        "comparison": comparison,
        "suite": arguments.suite,
        "cases": [
            case.to_dict() for case in case_module.SUITES[arguments.suite]
        ],
        "implementations": names,
        "scope": arguments.scope,
        "threads": arguments.threads,
        "seed": arguments.seed,
        "solver": {
            "algorithm": arguments.algorithm,
            "rtol": arguments.rtol,
            "atol": arguments.atol,
            "maxiter": arguments.maxiter,
            "residual_refresh": arguments.residual_refresh,
            "fused": arguments.fused,
            "preconditioner": arguments.preconditioner,
        },
    }
    if arguments.device == "cuda":
        spec.update(device="cuda", cuda_device=arguments.cuda_device)
    return spec


def main(argv=None):
    arguments = parser().parse_args(argv)
    if arguments.command == "_worker":
        return _run_worker(json.loads(arguments.spec_json), arguments.output)
    if arguments.command == "run":
        spec = benchmark_spec(arguments)
        spec["sampling"] = {
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
            "rounds": arguments.rounds,
        }
        return _launch(spec, arguments.output)
    from . import artifacts as artifact_module

    if arguments.command == "compare":
        artifacts = [
            artifact_module.load_artifact(path) for path in arguments.artifacts
        ]
        first = artifacts[0]
        for path, artifact in zip(
            arguments.artifacts, artifacts, strict=False
        ):
            print(f"Artifact: {path}")
            if (
                artifact["spec"] != first["spec"]
                or artifact["environment"] != first["environment"]
            ):
                print(
                    "Specification or environment differs; inspect metadata before comparing."
                )
            artifact_module.print_summary(artifact)
        return 0
    artifact = artifact_module.load_artifact(arguments.artifact)
    spec = dict(artifact["spec"])
    spec["expected_hashes"] = {
        case["case"]["name"]: case["input_sha256"]
        for case in artifact["cases"]
    }
    spec["expected_candidates"] = {
        case["case"]["name"]: [
            entry["name"]
            for entry in case["results"]
            if entry["status"] == "measured"
        ]
        for case in artifact["cases"]
    }
    spec["resolved_sampling"] = {
        case["case"]["name"]: case["sampling"] for case in artifact["cases"]
    }
    spec["expected_round_orders"] = {
        case["case"]["name"]: case["round_orders"]
        for case in artifact["cases"]
    }
    # Preserve each saved cycle count (the worker resolves the same candidate
    # set); unavailable candidates and versions remain visible in metadata.
    return _launch(spec, arguments.output)
