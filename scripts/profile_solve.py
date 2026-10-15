"""Diagnostic completed-wall profiling; never a framework comparison score.

Run in a fresh process with the selected staged package on PYTHONPATH. For CUDA:
nsys profile --capture-range=cudaProfilerApi --trace=cuda,cublas,nvtx \
    --sample=none --cpuctxsw=none .venv-cuda/bin/python \
    scripts/profile_solve.py ... --cuda-profiler
Input uploads, correctness downloads, warmups and environment capture are excluded.
"""

import argparse
import contextlib
import ctypes
import hashlib
import importlib.metadata
import os
import pathlib
import statistics
import sys
import time


def arguments():
    from cgbench.cli import IMPLEMENTATIONS, configure_benchmark_arguments

    parser = argparse.ArgumentParser(description=__doc__)
    configure_benchmark_arguments(parser)
    parser.add_argument(
        "--implementation", choices=IMPLEMENTATIONS, required=True
    )
    parser.add_argument(
        "--case", required=True, help="exact case name in --suite"
    )
    parser.add_argument(
        "--probe", choices=("solve", "workspace", "prepare"), default="solve"
    )
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--stage-identity", type=pathlib.Path)
    parser.add_argument("--cuda-profiler", action="store_true")
    args = parser.parse_args()
    if args.implementations is not None:
        parser.error("use the single --implementation option")
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be nonnegative and repetitions positive")
    if args.cuda_profiler and args.device != "cuda":
        parser.error("--cuda-profiler requires --device cuda")
    if args.probe != "solve" and args.implementation not in (
        "native",
        "native-blas",
        "native-cg",
        "native-pcg",
        "cuda-native",
        "cuda-cublas",
    ):
        parser.error(
            "workspace/prepare probes require a native public cgpy implementation"
        )
    args.implementations = args.implementation
    return args


def phase_call(execute, args, size):
    """Construct public resources; the caller drops their last reference in-range."""
    if args.probe == "solve":
        return execute
    if args.device == "cuda":
        from cgpy import cuda

        settings = dict(
            device=args.cuda_device, stream=0, dense_kernel=execute.kernel
        )
        if args.probe == "workspace":
            if hasattr(execute, "execution_mode"):
                settings["execution_mode"] = execute.execution_mode
            return lambda: cuda.Workspace(size, **settings)
        return lambda: cuda.prepare(
            execute.matrix, preconditioner=execute.preconditioner, **settings
        )
    import cgpy

    if args.probe == "workspace":
        return lambda: cgpy.Workspace(size)
    return lambda: cgpy.prepare(
        execute.matrix,
        num_threads=args.threads,
        dense_kernel=execute.dense_kernel,
        preconditioner=execute.preconditioner,
    )


def profiler_api(enabled, threads):
    """Use the public project-local NVIDIA wheel runtime, never a Torch-private DLL."""
    if not enabled:
        return None, None, None
    import torch
    from torch.cuda import nvtx

    torch.set_num_threads(threads)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)

    distribution = importlib.metadata.distribution("nvidia-cuda-runtime-cu12")
    path = pathlib.Path(
        distribution.locate_file("nvidia/cuda_runtime/lib/libcudart.so.12")
    ).resolve()
    if not path.is_file() or not path.is_relative_to(
        pathlib.Path(sys.prefix).resolve()
    ):
        raise RuntimeError(
            "--cuda-profiler requires libcudart in this local virtual environment"
        )
    library = ctypes.CDLL(str(path))
    for name in ("cudaProfilerStart", "cudaProfilerStop"):
        function = getattr(library, name)
        function.argtypes, function.restype = [], ctypes.c_int
    return (
        library,
        nvtx,
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        },
    )


@contextlib.contextmanager
def capture(library, device):
    if library is None:
        yield
        return
    import torch

    with torch.cuda.device(device):
        if error := library.cudaProfilerStart():
            raise RuntimeError(
                f"cudaProfilerStart failed with CUDA error {error}"
            )
        try:
            yield
        finally:
            if error := library.cudaProfilerStop():
                raise RuntimeError(
                    f"cudaProfilerStop failed with CUDA error {error}"
                )


@contextlib.contextmanager
def annotation(nvtx, label):
    if nvtx is not None:
        nvtx.range_push(label)
    try:
        yield
    finally:
        if nvtx is not None:
            nvtx.range_pop()


def stage_identity(path):
    if path is None:
        return None
    import json

    path = path.resolve()
    data = json.loads(path.read_text())
    checked = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(("cgbench", "cgpy")) or not getattr(
            module, "__file__", None
        ):
            continue
        binary = pathlib.Path(module.__file__).resolve()
        relative = binary.relative_to(path.parent).as_posix()
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        if data["files"].get(relative) != digest:
            raise RuntimeError(
                f"actual loaded module disagrees with stage identity: {binary}"
            )
        checked[name] = {"path": str(binary), "sha256": digest}
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "compiled_revision": data["compiled_revision"],
        "checked_modules": checked,
    }


def main():
    args = arguments()
    from cgbench import artifacts, cases, cli, collector, execution, json_io

    # This process must start fresh: set limits before any numerical imports.
    for name in artifacts.THREAD_VARIABLES:
        os.environ[name] = str(args.threads)
    os.environ.update(OMP_DYNAMIC="FALSE", MKL_DYNAMIC="FALSE")
    spec = cli.benchmark_spec(args)
    if args.device == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from cgbench.cuda_runtime import preflight_cuda

        preflight_cuda([args.implementation], args.cuda_device)
    selected = [entry for entry in spec["cases"] if entry["name"] == args.case]
    if len(selected) != 1:
        raise ValueError(
            f"--case must be one of {[entry['name'] for entry in spec['cases']]}"
        )
    case = cases.CaseSpec(**selected[0])
    fixture = cases.make_case(case, args.seed)
    arrays = fixture[:3]
    execute = execution.make_executor(
        args.implementation,
        arrays,
        spec["solver"],
        args.scope,
        args.threads,
        device=args.device,
        cuda_device=args.cuda_device,
    )
    phase = phase_call(execute, args, arrays[1].size)
    synchronize = getattr(execute, "synchronize", lambda: None)

    def gate():
        outcome = execute()
        if hasattr(execute, "host_outcome"):
            outcome = execute.host_outcome(outcome)
        result = collector.check_outcome(outcome, arrays, spec["solver"])
        if not result["eligible"]:
            raise RuntimeError(f"true-residual gate failed: {result}")
        return result

    library, nvtx, runtime = profiler_api(args.cuda_profiler, args.threads)
    before = gate()
    for _ in range(args.warmups):
        resource = phase()
        del resource
        synchronize()
    identity = stage_identity(args.stage_identity)
    elapsed = []
    synchronize()
    with capture(library, args.cuda_device):
        for index in range(args.repetitions):
            synchronize()
            with annotation(
                nvtx, f"{args.implementation}/{args.probe}/{index}"
            ):
                start = time.perf_counter_ns()
                resource = phase()
                del resource
                synchronize()
                elapsed.append(time.perf_counter_ns() - start)
    after = gate()
    if stage_identity(args.stage_identity) != identity:
        raise RuntimeError("stage identity changed during profiling")
    spec["cases"] = selected
    result = {
        "schema_version": 1,
        "metric": "diagnostic_phase_completed_wall",
        "framework_score": False,
        "spec": spec,
        "probe": args.probe,
        "warmups": args.warmups,
        "repetitions": args.repetitions,
        "phase_elapsed_ns": elapsed,
        "median_ns": statistics.median(elapsed),
        "input_sha256": cases.input_hash(case, fixture),
        "dtype": "float64",
        "before": before,
        "after": after,
        "adapter_setup": execute.benchmark_metadata(),
        "environment": artifacts.environment_metadata(
            device=args.device, cuda_device=args.cuda_device
        ),
        "stage_identity": identity,
        "profiler_runtime": runtime,
        "command": sys.argv,
        "scope": {
            "solve": (
                "session executor; fresh output/reset and per-call M; iteration workspace reused"
                if args.scope == "session"
                else "existing executor; x0 reset every call; api includes output/workspace/M allocation"
            ),
            "workspace": "public Workspace construction AND destruction; selected CUDA kernel resources",
            "prepare": "public prepare construction AND destruction; operator snapshot AND optional Jacobi",
        }[args.probe],
        "exclusions": "fixture generation, canonical CSR wrapper, host/device input uploads, correctness downloads, warmups, NVTX markers, profiler start/stop, metadata",
        "limits": "diagnostic single-process wall samples, includes host dispatch and completion fences; not independent framework samples or additive phase attribution",
    }
    json_io.write_json(result, args.output)
    print(
        f"Diagnostic {args.probe}: median {result['median_ns'] / 1e6:.6f} ms; saved {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
