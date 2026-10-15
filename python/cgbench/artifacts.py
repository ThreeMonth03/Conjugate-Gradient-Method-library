"""Store exact sampling data and report per-round average timings."""

import hashlib
import json
import os
import pathlib
import platform
import subprocess
import sys

from . import json_io, validation

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _command(arguments, cwd):
    try:
        completed = subprocess.run(
            arguments, cwd=cwd, capture_output=True, text=True, check=True
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_metadata(*, device="cpu", cuda_device=0):
    """Record actual library/thread/build information, with missing data explicit."""
    import importlib.metadata

    import numpy as np

    packages = {}
    for name in ("numpy", "scipy", "torch", "threadpoolctl", "cgpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    threadpools = None
    try:
        import threadpoolctl

        threadpools = threadpoolctl.threadpool_info()
    except ImportError:
        pass
    try:
        numpy_config = np.__config__.show(mode="dicts")
    except TypeError:
        numpy_config = {
            "information": "configuration unavailable with this NumPy"
        }
    cmake = None
    native = None
    source_root = None
    native_module = sys.modules.get("cgpy._core")
    if native_module is not None and getattr(native_module, "__file__", None):
        binary = pathlib.Path(native_module.__file__).resolve()
        package = sys.modules.get("cgpy")
        build_info = getattr(package, "build_info", None)
        native = {
            "module_path": str(binary),
            "module_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "build_info": build_info() if callable(build_info) else None,
            "python_package_path": getattr(package, "__file__", None),
        }
        # CMake stages extensions under <binary-dir>/python/cgpy. A wheel in a
        # repository's virtual environment must not borrow an ancestor cache.
        for directory in binary.parents:
            if not binary.parent.is_relative_to(directory / "python/cgpy"):
                continue
            cache = directory / "CMakeCache.txt"
            if not cache.is_file():
                continue
            fields = {}
            for line in cache.read_text(errors="replace").splitlines():
                if line.startswith(
                    (
                        "CMAKE_CXX_COMPILER:",
                        "CMAKE_CXX_FLAGS",
                        "CMAKE_BUILD_TYPE:",
                        "CMAKE_HOME_DIRECTORY:",
                        "CG_",
                    )
                ):
                    key, _, value = line.partition("=")
                    fields[key] = value
            cmake = {"path": str(cache), "fields": fields}
            home = fields.get("CMAKE_HOME_DIRECTORY:INTERNAL")
            if home and pathlib.Path(home).is_absolute():
                source_root = pathlib.Path(home).resolve()
            break
    cpu = platform.processor() or platform.machine()
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                cpu = line.partition(":")[2].strip()
                break
    metadata = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
        "python": sys.version,
        "python_executable": sys.executable,
        "python_compiler": platform.python_compiler(),
        "packages": packages,
        "source_root": str(source_root) if source_root is not None else None,
        "revision": _command(["git", "rev-parse", "HEAD"], source_root)
        if source_root is not None
        else None,
        "git_status": _command(["git", "status", "--short"], source_root)
        if source_root is not None
        else None,
        "cmake": cmake,
        "native": native,
        "thread_environment": {
            name: os.environ.get(name) for name in THREAD_VARIABLES
        },
        "dynamic_thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_DYNAMIC", "MKL_DYNAMIC")
        },
        "threadpools": threadpools,
        "numpy_config": numpy_config,
        "torch": None,
    }
    if "torch" in sys.modules:
        import torch

        metadata["torch"] = {
            "intraop_threads": torch.get_num_threads(),
            "interop_threads": torch.get_num_interop_threads(),
            "config": torch.__config__.show(),
        }
    if device == "cuda":
        from . import cuda_metadata

        metadata["cuda"] = cuda_metadata.cuda_metadata(cuda_device)
    return metadata


def validate_artifact(data):
    if data.get("schema_version") != 1 or not isinstance(
        data.get("cases"), list
    ):
        raise ValueError("unsupported benchmark artifact")
    spec = data.get("spec", {})
    device = spec.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError("artifact device must be cpu or cuda")
    validation.validate_execution_mode(spec.get("solver", {}), device)
    if device == "cuda":
        from . import execution

        ordinal = spec.get("cuda_device")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError(
                "artifact CUDA device must be a nonnegative integer"
            )
        if any(
            name not in execution.CUDA_IMPLEMENTATIONS
            for name in spec.get("implementations", [])
        ):
            raise ValueError("CUDA artifact cannot contain CPU candidates")
        if spec.get("scope") not in ("api", "prepared"):
            raise ValueError(
                "CUDA artifact must use an explicit resident scope"
            )
    for case in data["cases"]:
        sampling = case["sampling"]
        rounds, repetitions = sampling["rounds"], sampling["repetitions"]
        if rounds < 1 or repetitions < 1 or rounds % case["cycle_rows"]:
            raise ValueError("invalid sampling or incomplete Williams cycle")
        if len(case["round_orders"]) != rounds:
            raise ValueError("artifact is missing timing rounds")
        measured = {
            entry["name"]
            for entry in case["results"]
            if entry["status"] == "measured"
        }
        for order in case["round_orders"]:
            # Failed post-measurement checks remain present in the saved schedule.
            expected = measured | {
                entry["name"]
                for entry in case["results"]
                if entry["status"] == "invalid"
                and entry.get("reason", "").startswith("post-measurement")
            }
            if len(order) != len(expected) or set(order) != expected:
                raise ValueError(
                    "round order does not match measured candidates"
                )
        for entry in case["results"]:
            elapsed = entry["round_elapsed_ns"]
            if entry["status"] == "measured":
                validation.validate_converged_result(entry)
                if len(elapsed) != rounds or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in elapsed
                ):
                    raise ValueError("invalid raw nanosecond samples")
            elif entry["status"] not in ("invalid", "ineligible") or elapsed:
                raise ValueError("invalid candidate status")
    return data


def write_artifact(data, output):
    validate_artifact(data)
    json_io.write_json(data, output)


def load_artifact(path):
    return validate_artifact(
        json.loads(pathlib.Path(path).read_text(encoding="utf8"))
    )


def summarize(data):
    import numpy as np

    rows = []
    for case in data["cases"]:
        for entry in case["results"]:
            row = {
                "case": case["case"]["name"],
                "implementation": entry["name"],
                "status": entry["status"],
                "setup": entry.get("setup", {}),
            }
            if entry["status"] == "measured":
                samples = (
                    np.asarray(entry["round_elapsed_ns"], dtype=np.float64)
                    / case["sampling"]["repetitions"]
                )
                p5, p25, median, p75, p95 = np.percentile(
                    samples, [5, 25, 50, 75, 95]
                )
                row.update(
                    median_ns=float(median),
                    p5_ns=float(p5),
                    p25_ns=float(p25),
                    p75_ns=float(p75),
                    p95_ns=float(p95),
                    iterations=entry["iterations"],
                    matvecs=entry["matvecs"],
                    true_residual_norm=entry["true_residual_norm"],
                )
            else:
                row["reason"] = entry["reason"]
            rows.append(row)
    return rows


def print_summary(data):
    rows = summarize(data)
    if data.get("metric") == "sparse_solver_comparison":
        print(
            "case solver status median_ms p5_ms p95_ms iterations matvecs true_residual setup_ms factor_stored_bytes"
        )
        for row in rows:
            if row["status"] != "measured":
                print(
                    f"{row['case']} {row['implementation']} {row['status']}: {row['reason']}"
                )
                continue
            setup = row["setup"]
            setup_ns = setup.get(
                "factorization_setup_elapsed_ns",
                setup.get("preconditioner_setup_elapsed_ns"),
            )
            setup_ms = (
                f"{setup_ns / 1e6:.6f}" if setup_ns is not None else "n/a"
            )
            print(
                f"{row['case']} {row['implementation']} measured {row['median_ns'] / 1e6:.6f} {row['p5_ns'] / 1e6:.6f} {row['p95_ns'] / 1e6:.6f} {row['iterations']} {row['matvecs']} {row['true_residual_norm']:.6e} {setup_ms} {setup.get('factor_stored_bytes', 0)}"
            )
        print(
            "Solver strategies differ: compare verified solution time and disclosed setup; no implementation speedup ratio."
        )
        print(
            "Setup samples are separate untimed probes; API preconditioning and LU-total factorization are also included in timed calls."
        )
        print(
            "Stored input/factor/preconditioner array bytes are not peak process memory. Percentiles summarize round-average latency."
        )
        return
    baseline = (
        "cuda-native"
        if data.get("spec", {}).get("device", "cpu") == "cuda"
        else "native"
    )
    native = {
        row["case"]: row["median_ns"]
        for row in rows
        if row["implementation"] == baseline and row["status"] == "measured"
    }
    ratio_name = (
        "cuda_native_speedup"
        if baseline == "cuda-native"
        else "native_speedup"
    )
    print(
        "case implementation status median_ms p5_ms p95_ms iterations matvecs "
        + ratio_name
    )
    for row in rows:
        if row["status"] == "measured":
            native_time = native.get(row["case"])
            speedup = (
                f"{row['median_ns'] / native_time:.3f}x"
                if native_time
                else "n/a"
            )
            print(
                f"{row['case']} {row['implementation']} measured {row['median_ns'] / 1e6:.6f} {row['p5_ns'] / 1e6:.6f} {row['p95_ns'] / 1e6:.6f} {row['iterations']} {row['matvecs']} {speedup}"
            )
        else:
            print(
                f"{row['case']} {row['implementation']} {row['status']}: {row['reason']}"
            )
    print(
        "Percentiles describe per-round average ns/call, not individual-call latency."
    )
    print(
        f"{ratio_name} = this implementation's median / {baseline} median; >1 means {baseline} is faster."
    )
