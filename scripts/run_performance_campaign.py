"""Run immutable, balanced CPU/CUDA campaigns over already built local stages.

The JSON input declares both frozen payloads and every configuration. Two time
windows each execute ABBA and BAAB in fresh interpreters; the second reverses
configuration order. Boundary telemetry is diagnostic and never corrects times.
The original comparator, its 5% budget and every framework control are retained.
Use --plan-only first, then the identical command without --plan-only. Started
campaigns cannot resume: partial evidence is retained in the progress manifest.
"""

import argparse
import copy
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter

THREADS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SIDES = ("baseline", "current")
QUARTETS = (
    ("baseline", "current", "current", "baseline"),
    ("current", "baseline", "baseline", "current"),
)
POLICY = {
    "windows": 2,
    "quartets_per_window": ["ABBA", "BAAB"],
    "artifact_blocks_per_side": 8,
    "max_regression": 0.05,
    "confidence": 0.95,
    "bootstrap_samples": 10000,
    "seed": 20261018,
    "telemetry": "before/after each child; no continuous observer",
    "no_retry_no_discard": True,
}


def sha(path):
    with pathlib.Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_new(path, value):
    """Create evidence once; an existing result is never replaced."""
    with pathlib.Path(path).open("x", encoding="utf8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def checkpoint(path, value):
    temporary = path.with_suffix(".json.tmp")
    write_new(temporary, value)
    temporary.replace(path)


def positive(value, label, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(
            f"{label} must be a {'nonnegative' if zero else 'positive'} integer"
        )
    return value


def payload(stage):
    """Validate the entire staged Python/native payload, not only one module."""
    root = pathlib.Path(stage["package_root"])
    if not root.is_absolute() or not root.is_dir() or root.resolve() != root:
        raise ValueError(
            "package_root must be an absolute, resolved directory"
        )
    for key in ("compiled_revision", "compiled_tree"):
        if not re.fullmatch("[0-9a-f]{40}", stage[key]):
            raise ValueError(f"stage requires full {key}")
    files = stage["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("stage files must declare absolute payload hashes")
    for name, digest in files.items():
        path = pathlib.Path(name)
        if (
            not path.is_absolute()
            or path.resolve() != path
            or not re.fullmatch("[0-9a-f]{64}", digest)
            or sha(path) != digest
        ):
            raise ValueError(f"frozen stage file differs: {name}")
    required = {
        str(path)
        for path in root.rglob("*")
        if path.is_file()
        and (
            path.suffix in {".py", ".pyi", ".so", ".pyd"}
            or path.name == "py.typed"
        )
    }
    if (
        not required.issubset(files)
        or not (root / "cgpy/__init__.py").is_file()
        or not (root / "cgbench/__main__.py").is_file()
    ):
        raise ValueError(
            "stage manifest omits required Python/native payload files"
        )
    return {
        str(pathlib.Path(name).relative_to(root)): digest
        for name, digest in files.items()
        if pathlib.Path(name).is_relative_to(root)
    }


def public_arguments(spec):
    """Serialize the shared public cgbench controls, then audit the resolved spec."""
    required = {
        "comparison",
        "suite",
        "cases",
        "implementations",
        "scope",
        "threads",
        "seed",
        "solver",
        "sampling",
    }
    if not required.issubset(spec) or set(spec) - required - {
        "device",
        "cuda_device",
    }:
        raise ValueError(
            "configuration must declare the full public benchmark spec"
        )
    device = spec.get("device", "cpu")
    if device == "cpu" and ("device" in spec or "cuda_device" in spec):
        raise ValueError("canonical CPU specs omit device/cuda_device keys")
    if device not in {"cpu", "cuda"} or spec["scope"] not in {
        "api",
        "session",
        "prepared",
    }:
        raise ValueError("unsupported benchmark device/scope")
    if device == "cpu" and spec["scope"] == "session":
        raise ValueError("session scope requires CUDA")
    names = spec["implementations"]
    if (
        not isinstance(names, list)
        or len(names) != len(set(names))
        or len(names) < 2
    ):
        raise ValueError(
            "retain unique native implementations and framework controls"
        )
    sparse = all(
        case.get("storage", "dense") == "csr" for case in spec["cases"]
    )
    native_names = (
        {"cuda-native", "cuda-cublas"}
        if device == "cuda"
        else {"native", "native-blas", "native-cg", "native-pcg"}
    )
    if not native_names.intersection(names):
        raise ValueError(
            "campaign requires a native candidate for the selected device"
        )
    controls = (
        {"torch-cuda-eager"}
        if device == "cuda"
        else {"scipy-out" if sparse else "numpy-out", "torch-eager-out"}
    )
    if not controls.issubset(names):
        raise ValueError(
            "campaign must retain every declared NumPy/PyTorch control"
        )
    solver, sampling = spec["solver"], spec["sampling"]
    if "execution_mode" in solver and (
        device != "cuda" or solver["execution_mode"] != "device"
    ):
        raise ValueError(
            "canonical host specs omit execution_mode; only CUDA device mode adds it"
        )
    if (
        solver["algorithm"] not in {"standard", "explicit"}
        or solver["preconditioner"] not in {"none", "jacobi"}
        or (
            solver["algorithm"] == "explicit"
            and solver["preconditioner"] != "none"
        )
    ):
        raise ValueError(
            "unsupported explicit algorithm/preconditioner combination"
        )
    required_solver = {
        "algorithm",
        "rtol",
        "atol",
        "maxiter",
        "residual_refresh",
        "fused",
        "preconditioner",
    }
    if set(solver) - required_solver - {
        "execution_mode"
    } or not required_solver.issubset(solver):
        raise ValueError("solver specification is incomplete")
    if (
        set(sampling) != {"warmups", "rounds", "repetitions"}
        or type(solver["fused"]) is not bool
    ):
        raise ValueError("sampling/fusion specification is invalid")
    for key in ("threads", "seed"):
        positive(spec[key], key, zero=key == "seed")
    for key in sampling:
        positive(sampling[key], key, zero=key == "warmups")
    cycle = len(names) if len(names) % 2 == 0 else 2 * len(names)
    if sampling["rounds"] % cycle:
        raise ValueError("rounds must cover complete Williams cycles")
    for key in ("maxiter", "residual_refresh"):
        positive(solver[key], key, zero=True)
    for key in ("rtol", "atol"):
        if (
            isinstance(solver[key], bool)
            or not isinstance(solver[key], (int, float))
            or not math.isfinite(solver[key])
            or solver[key] < 0
        ):
            raise ValueError(f"invalid {key}")
    args = [
        "--suite",
        spec["suite"],
        "--implementations",
        ",".join(names),
        "--scope",
        spec["scope"],
        "--threads",
        str(spec["threads"]),
        "--seed",
        str(spec["seed"]),
    ]
    for key in (
        "algorithm",
        "rtol",
        "atol",
        "maxiter",
        "residual_refresh",
        "preconditioner",
    ):
        args += ["--" + key.replace("_", "-"), str(solver[key])]
    args.append("--fused" if solver["fused"] else "--no-fused")
    for key, value in sampling.items():
        args += ["--" + key, str(value)]
    if device == "cuda":
        positive(spec.get("cuda_device"), "cuda_device", zero=True)
        args += [
            "--device",
            "cuda",
            "--cuda-device",
            str(spec["cuda_device"]),
            "--cuda-execution-mode",
            solver.get("execution_mode", "host"),
        ]
    return args


def make_plan(input_path, output):
    """Freeze a reproducible schedule before any timing outcome is available."""
    source = json.loads(input_path.read_text(encoding="utf8"))
    if source.get("schema_version") != 1 or source.get("mode") not in {
        "calibration",
        "comparison",
    }:
        raise ValueError(
            "require schema_version 1 and calibration/comparison mode"
        )
    if set(source["stages"]) != set(SIDES):
        raise ValueError("require both baseline and current stages")
    content = {side: payload(source["stages"][side]) for side in SIDES}
    if (
        source["mode"] == "calibration"
        and content["baseline"] != content["current"]
    ):
        raise ValueError(
            "A/A calibration requires identical staged payload bytes"
        )
    native = {
        side: {
            name: digest
            for name, digest in content[side].items()
            if pathlib.Path(name).suffix in {".so", ".pyd"}
        }
        for side in SIDES
    }
    if not native["baseline"] or not native["current"]:
        raise ValueError("stages must contain frozen native extensions")
    if (
        source["mode"] == "comparison"
        and native["baseline"] == native["current"]
    ):
        raise ValueError(
            "comparison requires a changed native payload; use calibration for A/A"
        )
    wrappers = {
        side: {
            name: digest
            for name, digest in content[side].items()
            if pathlib.Path(name).suffix in {".py", ".pyi"}
            or name.endswith("py.typed")
        }
        for side in SIDES
    }
    if wrappers["baseline"] != wrappers["current"]:
        raise ValueError(
            "both sides require identical Python/reference benchmark code"
        )
    comparator = pathlib.Path(
        source.get(
            "comparator",
            pathlib.Path(__file__).with_name("compare_performance.py"),
        )
    ).resolve(strict=True)
    pause = source.get("window_pause_seconds", 60)
    timeout = source.get("worker_timeout_seconds", 600)
    for value, label in (
        (pause, "window_pause_seconds"),
        (timeout, "worker_timeout_seconds"),
    ):
        if (
            type(value) not in {int, float}
            or not math.isfinite(value)
            or value < (0 if label.startswith("window") else 1)
        ):
            raise ValueError(f"invalid {label}")
    configs = source["configurations"]
    if not configs or len({row["name"] for row in configs}) != len(configs):
        raise ValueError("require nonempty, uniquely named configurations")
    if len({row["preset"] for row in configs}) != 1:
        raise ValueError(
            "one frozen package pair supports exactly one preset per campaign"
        )
    executables, runs = {}, []
    for row in configs:
        if not re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", row["name"]):
            raise ValueError(
                "configuration name must be a safe unique filename"
            )
        public_arguments(row["spec"])
        affinity, threads = row["affinity"], row["spec"]["threads"]
        expected_threads = 2 if row["preset"] == "cpu-openmp" else 1
        if (
            row["preset"]
            not in {"cpu-release", "cpu-blas", "cpu-openmp", "cuda-release"}
            or threads != expected_threads
            or not isinstance(affinity, list)
            or len(affinity) != threads
            or len(set(affinity)) != threads
            or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
            or (row["preset"] == "cuda-release")
            != (row["spec"].get("device") == "cuda")
        ):
            raise ValueError(
                "preset, thread count, device or affinity differs"
            )
        if affinity != sorted(affinity):
            raise ValueError("affinity must be sorted in ascending CPU order")
        names = row["spec"]["implementations"]
        dense = all(
            case.get("storage", "dense") == "dense"
            for case in row["spec"]["cases"]
        )
        required_candidates = (
            {"cuda-native", "cuda-cublas"}
            if row["preset"] == "cuda-release" and dense
            else {"cuda-native"}
            if row["preset"] == "cuda-release"
            else {"native", "native-blas"}
            if row["preset"] == "cpu-blas" and dense
            else {"native"}
        )
        if not required_candidates.issubset(names):
            raise ValueError(
                "preset must retain every applicable native backend candidate"
            )
        python = pathlib.Path(row["python"])
        if not python.is_absolute() or not python.is_file():
            raise ValueError(
                "configuration python must be an existing absolute interpreter"
            )
        executables[str(python)] = sha(python)
        for side in SIDES:
            modules = [
                name
                for name in content[side]
                if re.fullmatch(r"cgpy/_core(?:\.[^/]+)?\.(?:so|pyd)", name)
            ]
            cuda_modules = [
                name
                for name in content[side]
                if re.fullmatch(r"cgpy/_cuda(?:\.[^/]+)?\.(?:so|pyd)", name)
            ]
            if len(modules) != 1 or (
                row["preset"] == "cuda-release" and len(cuda_modules) != 1
            ):
                raise ValueError(
                    "stage requires one _core extension and CUDA presets one _cuda extension"
                )
        if source["mode"] == "comparison":
            prefix = (
                "cgpy/_cuda"
                if row["preset"] == "cuda-release"
                else "cgpy/_core"
            )
            active = {
                side: [
                    digest
                    for name, digest in content[side].items()
                    if name.startswith(prefix)
                    and pathlib.Path(name).suffix in {".so", ".pyd"}
                ]
                for side in SIDES
            }
            if active["baseline"] == active["current"]:
                raise ValueError(
                    "comparison requires a changed active native extension"
                )
    for window in (1, 2):
        for row in configs if window == 1 else reversed(configs):
            for quartet, order in enumerate(QUARTETS, 1):
                for slot, side in enumerate(order, 1):
                    name = f"w{window}-q{quartet}-s{slot}-{row['name']}-{side}"
                    raw = output / "raw" / (name + ".json")
                    runs.append(
                        {
                            "id": name,
                            "configuration": row["name"],
                            "window": window,
                            "quartet": quartet,
                            "slot": slot,
                            "side": side,
                            "output": str(raw),
                            "log": str(raw.with_suffix(".log")),
                            "argv": [
                                "taskset",
                                "-c",
                                ",".join(map(str, row["affinity"])),
                                row["python"],
                                "-m",
                                "cgbench",
                                "run",
                                *public_arguments(row["spec"]),
                                "--output",
                                str(raw),
                            ],
                        }
                    )
    return {
        "schema_version": 1,
        "input_plan": str(input_path),
        "input_sha256": sha(input_path),
        "source": source,
        "policy": POLICY,
        "window_pause_seconds": pause,
        "worker_timeout_seconds": timeout,
        "executables": executables,
        "harness": {
            str(pathlib.Path(__file__).resolve()): sha(__file__),
            str(comparator): sha(comparator),
        },
        "comparator": str(comparator),
        "runs": runs,
    }


def prepare_output(plan, output, plan_only):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        saved = json.loads(path.read_text())
        manifest = json.loads((output / "manifest.json").read_text())
        if (
            plan_only
            or saved != plan
            or manifest.get("status") != "planned"
            or manifest.get("plan_sha256") != sha(path)
            or set(manifest)
            != {"status", "planned_utc", "plan_sha256", "runs"}
            or not isinstance(manifest.get("planned_utc"), str)
            or manifest.get("runs") != []
            or set(output.iterdir()) != {path, output / "manifest.json"}
        ):
            raise ValueError(
                "only an identical pristine plan may execute; no resume or overwrite"
            )
        return manifest
    if any(output.iterdir()):
        raise ValueError("output directory must be empty")
    write_new(path, plan)
    manifest = {
        "status": "planned",
        "planned_utc": now(),
        "plan_sha256": sha(path),
        "runs": [],
    }
    write_new(output / "manifest.json", manifest)
    return manifest


def telemetry():
    """Observe boundaries outside child execution; unsupported sensors stay explicit."""
    result = {"utc": now(), "monotonic_ns": time.monotonic_ns(), "files": {}}
    paths = [
        pathlib.Path(name)
        for name in ("/proc/loadavg", "/proc/stat", "/proc/pressure/cpu")
    ]
    paths += sorted(
        pathlib.Path("/sys/devices/system/cpu").glob(
            "cpu*/cpufreq/scaling_cur_freq"
        )
    )
    paths += sorted(
        pathlib.Path("/sys/class/thermal").glob("thermal_zone*/temp")
    )
    for path in paths:
        try:
            result["files"][str(path)] = {
                "value": path.read_text().strip(),
                "error": None,
            }
        except OSError as error:
            result["files"][str(path)] = {"value": None, "error": str(error)}
    for label, pattern in (
        ("cpu_frequency", "cpufreq/scaling_cur_freq"),
        ("cpu_temperature", "/temp"),
    ):
        result[label] = (
            "available"
            if any(pattern in str(path) for path in paths)
            else "unavailable"
        )
    try:
        mhz = [
            line
            for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("cpu MHz")
        ]
        result["cpu_mhz"] = {
            "value": mhz or None,
            "error": None if mhz else "unavailable",
        }
    except OSError as error:
        result["cpu_mhz"] = {"value": None, "error": str(error)}
    executable = shutil.which("nvidia-smi")
    if (
        executable is None
        and pathlib.Path("/usr/lib/wsl/lib/nvidia-smi").is_file()
    ):
        executable = "/usr/lib/wsl/lib/nvidia-smi"
    result["gpu"] = {
        "available": executable is not None,
        "argv": None,
        "returncode": None,
        "stdout": None,
        "stderr": None,
        "error": None,
    }
    if executable:
        command = [
            executable,
            "--query-gpu=uuid,utilization.gpu,utilization.memory,temperature.gpu,clocks.sm,clocks.mem,power.draw,pstate",
            "--format=csv,noheader,nounits",
        ]
        result["gpu"]["argv"] = command
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            result["gpu"].update(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except (OSError, subprocess.SubprocessError) as error:
            result["gpu"]["error"] = f"{type(error).__name__}: {error}"
    else:
        result["gpu"]["error"] = "nvidia-smi unavailable"
    return result


def environment(stage, spec):
    env = dict(
        os.environ,
        PYTHONPATH=stage["package_root"],
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_DYNAMIC="FALSE",
        MKL_DYNAMIC="FALSE",
    )
    env.update({name: str(spec["threads"]) for name in THREADS})
    for name in ("PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(name, None)
    if spec.get("device") == "cuda":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env.pop("CUDA_LAUNCH_BLOCKING", None)
    return env


def audit_setup(entry, spec, build):
    """Require actual adapter metadata to preserve each allocation/mode contract."""
    name, setup, solver = entry["name"], entry["setup"], spec["solver"]
    preconditioner = (
        "jacobi"
        if name.endswith("-pcg")
        else "none"
        if name.endswith("-cg")
        else solver["preconditioner"]
    )
    if setup.get("preconditioner") != preconditioner:
        raise ValueError("actual adapter preconditioner differs")
    if setup.get("setup_in_timed_call") is not (
        spec["scope"] in ("api", "session")
    ):
        raise ValueError("actual setup timing scope differs")
    if setup.get("residual_policy") != solver[
        "algorithm"
    ] + " with residual_refresh=" + str(solver["residual_refresh"]):
        raise ValueError("actual adapter algorithm/refresh policy differs")
    if spec.get("device", "cpu") == "cpu":
        if name.startswith("native"):
            kernel = "blas" if name == "native-blas" else "native"
            if setup.get("dense_kernel") != kernel or (
                kernel == "blas" and setup.get("blas") != build["blas"]
            ):
                raise ValueError("actual native CPU kernel/provider differs")
        return
    mode = solver.get("execution_mode", "host")
    if (
        setup.get("cuda_device") != spec["cuda_device"]
        or setup.get("stream") != 0
    ):
        raise ValueError("actual CUDA device/stream differs")
    if name == "torch-cuda-eager":
        if (
            setup.get("native_execution_mode") != mode
            or setup.get("native_execution_mode_applies") is not False
        ):
            raise ValueError(
                "Torch must disclose ignoring the native execution mode"
            )
    elif (
        setup.get("execution_mode") != mode
        or setup.get("fused") is not solver["fused"]
        or setup.get("dense_kernel")
        != ("cublas" if name == "cuda-cublas" else "native")
    ):
        raise ValueError("actual explicit CUDA kernel/fusion/mode differs")
    if spec["scope"] == "session":
        contract = {
            "workspace_reused": True,
            "workspace_in_timed_call": False,
            "output_reused": False,
            "output_in_timed_call": True,
            "preconditioner_reused": False,
            "preconditioner_in_timed_call": solver["preconditioner"]
            == "jacobi",
        }
        if any(setup.get(key) is not value for key, value in contract.items()):
            raise ValueError(
                "Session fresh-output/M/reused-workspace contract differs"
            )


def audit_raw(data, config, stage):
    """Reject mismatched controls before a row can reach the strict comparator."""
    spec, env = config["spec"], data["environment"]
    if data.get("schema_version") != 1 or data["spec"] != spec:
        raise ValueError("actual full benchmark specification differs")
    if [case["case"] for case in data["cases"]] != spec["cases"]:
        raise ValueError("actual fixture inventory differs")
    for case in data["cases"]:
        results = case["results"]
        names = spec["implementations"]
        cycle = len(names) if len(names) % 2 == 0 else 2 * len(names)
        rounds = spec["sampling"]["rounds"]
        if (
            case["dtype"] != "float64"
            or case["sampling"] != spec["sampling"]
            or case["schedule"] != "williams"
            or case["cycle_rows"] != cycle
            or rounds % cycle
            or len(case["round_orders"]) != rounds
            or not re.fullmatch("[0-9a-f]{64}", case["input_sha256"])
            or [entry["name"] for entry in results] != names
            or any(entry["status"] != "measured" for entry in results)
        ):
            raise ValueError(
                "actual result/sampling inventory omitted or invalidated a candidate/control"
            )
        orders = case["round_orders"]
        if any(
            len(order) != len(names) or set(order) != set(names)
            for order in orders
        ):
            raise ValueError("incomplete Williams row")
        positions = Counter(
            (name, index)
            for order in orders
            for index, name in enumerate(order)
        )
        predecessors = Counter(
            (left, right)
            for order in orders
            for left, right in zip(order, order[1:], strict=False)
        )
        if (
            len(positions) != len(names) ** 2
            or len(set(positions.values())) != 1
            or len(predecessors) != len(names) * (len(names) - 1)
            or len(set(predecessors.values())) != 1
        ):
            raise ValueError("unbalanced Williams positions/predecessors")
        for entry in results:
            audit_setup(entry, spec, env["native"]["build_info"])
            residual, threshold = (
                entry["true_residual_norm"],
                entry["residual_threshold"],
            )
            if (
                entry["solver_status"] != "converged"
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0
                    for value in (residual, threshold)
                )
                or residual > threshold * (1 + 64 * sys.float_info.epsilon)
            ):
                raise ValueError(
                    "independent true-residual convergence gate failed"
                )
            if (
                any(
                    type(entry.get(key)) is not int or entry[key] < 0
                    for key in ("iterations", "matvecs", "restarts")
                )
                or entry["restarts"] > entry["iterations"]
                or (
                    spec["solver"]["algorithm"] == "explicit"
                    and entry["restarts"] != 0
                )
            ):
                raise ValueError(
                    "known valid iterations/matvecs/restarts are required"
                )
            if len(entry["round_elapsed_ns"]) != rounds or any(
                type(value) is not int or value <= 0
                for value in entry["round_elapsed_ns"]
            ):
                raise ValueError("invalid raw elapsed nanosecond samples")
    threads = spec["threads"]
    if (
        env["affinity"] != config["affinity"]
        or env["python_executable"] != config["python"]
        or env["thread_environment"]
        != {name: str(threads) for name in THREADS}
        or env["dynamic_thread_environment"]
        != {"OMP_DYNAMIC": "FALSE", "MKL_DYNAMIC": "FALSE"}
        or not env["threadpools"]
        or any(pool["num_threads"] != threads for pool in env["threadpools"])
        or env["torch"]["intraop_threads"] != threads
        or env["torch"]["interop_threads"] != 1
    ):
        raise ValueError(
            "actual interpreter, affinity or runtime thread controls differ"
        )
    build = env["native"]["build_info"]
    if bool(build["openmp"]) != (config["preset"] == "cpu-openmp") or bool(
        build["blas"]["enabled"]
    ) != (config["preset"] == "cpu-blas"):
        raise ValueError("actual native build differs from declared preset")
    if (
        pathlib.Path(env["native"]["python_package_path"]).resolve()
        != pathlib.Path(stage["package_root"]) / "cgpy/__init__.py"
    ):
        raise ValueError("loaded wrapper is outside the frozen stage")
    modules = [env["native"]]
    signature = {
        key: env.get(key)
        for key in (
            "platform",
            "machine",
            "cpu",
            "logical_cpus",
            "python",
            "python_executable",
            "python_compiler",
            "packages",
            "affinity",
            "thread_environment",
            "dynamic_thread_environment",
            "threadpools",
            "numpy_config",
            "torch",
        )
    }
    signature["native_build"] = build
    signature["compiler_flags"] = {
        name.partition(":")[0]: value
        for name, value in (env.get("cmake") or {}).get("fields", {}).items()
        if name.startswith(
            ("CMAKE_CXX_COMPILER:", "CMAKE_CXX_FLAGS", "CMAKE_BUILD_TYPE:")
        )
    }
    if spec.get("device") == "cuda":
        cuda = env["cuda"]
        if (
            cuda["device"] != spec["cuda_device"]
            or cuda["environment"]["CUBLAS_WORKSPACE_CONFIG"] != ":4096:8"
            or cuda["environment"]["CUDA_LAUNCH_BLOCKING"] is not None
        ):
            raise ValueError("actual CUDA device or timing controls differ")
        modules.append(cuda["native"])
        signature["cuda"] = {
            name: cuda.get(name)
            for name in ("device", "environment", "nvidia_smi", "torch")
        }
        signature["cuda_native"] = {
            name: cuda["native"].get(name)
            for name in ("runtime_info", "build_info")
        }
    for module in modules:
        path = pathlib.Path(module["module_path"])
        if (
            not path.is_relative_to(stage["package_root"])
            or stage["files"].get(str(path)) != module["module_sha256"]
        ):
            raise ValueError(
                "loaded native binary differs from frozen payload"
            )
    return signature


def verify_frozen(plan):
    for name, digest in {
        plan["input_plan"]: plan["input_sha256"],
        **plan["harness"],
        **plan["executables"],
    }.items():
        if sha(name) != digest:
            raise ValueError(f"frozen plan/tool/interpreter changed: {name}")
    for stage in plan["source"]["stages"].values():
        payload(stage)


def terminate_group(process):
    """Stop workers even when their launcher has already exited."""
    try:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    finally:
        # A dead launcher or interrupted grace period does not establish that
        # all workers exited; they may have ignored SIGTERM.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_command(command, *, cwd, env, log, timeout):
    """Stop the complete group after failed, timed-out or interrupted children."""
    with pathlib.Path(log).open("x", encoding="utf8") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout)
            if returncode:
                terminate_group(process)
            return returncode
        except BaseException:
            terminate_group(process)
            raise


def compare_all(plan, manifest, output):
    """Keep complete primary reports unchanged; controls never rescue a failure."""
    reports = []
    for config in plan["source"]["configurations"]:
        runs = [
            run
            for run in manifest["runs"]
            if run["configuration"] == config["name"]
        ]
        paths = {
            side: [run["output"] for run in runs if run["side"] == side]
            for side in SIDES
        }
        if any(
            len(values) != POLICY["artifact_blocks_per_side"]
            for values in paths.values()
        ):
            raise ValueError("strict comparison requires every fresh block")
        target = output / "reports" / (config["name"] + ".json")
        command = [
            config["python"],
            plan["comparator"],
            "--baseline",
            *paths["baseline"],
            "--current",
            *paths["current"],
            "--max-regression",
            "0.05",
            "--bootstrap-samples",
            "10000",
            "--seed",
            str(POLICY["seed"]),
            "--output",
            str(target),
        ]
        if target.exists():
            raise ValueError("comparison output already exists")
        returncode = run_command(
            command,
            cwd=output,
            env=environment(
                plan["source"]["stages"]["current"], config["spec"]
            ),
            log=target.with_suffix(".log"),
            timeout=plan["worker_timeout_seconds"],
        )
        report = json.loads(target.read_text())
        expected_metric = (
            "sparse_solver_comparison"
            if config["spec"]["comparison"] == "sparse-solvers"
            else "time_to_solution"
        )
        if (
            returncode not in {0, 1}
            or report["max_regression"] != 0.05
            or report["ratio_limit"] != 1.05
            or report["baseline_files"] != paths["baseline"]
            or report["current_files"] != paths["current"]
            or report["bootstrap"]["resamples"] != 10000
            or report["bootstrap"]["seed"] != POLICY["seed"]
            or report["bootstrap"]["confidence"] != 0.95
            or report["restart_parity_verified"] is not True
            or report["benchmark"]
            != {
                "metric": expected_metric,
                "spec": {
                    key: value
                    for key, value in config["spec"].items()
                    if key != "sampling"
                },
            }
            or {
                (row["case"]["name"], row["implementation"])
                for row in report["cases"]
            }
            != {
                (case["name"], name)
                for case in config["spec"]["cases"]
                for name in config["spec"]["implementations"]
            }
            or len(report["cases"])
            != len(config["spec"]["cases"])
            * len(config["spec"]["implementations"])
            or returncode
            != int(report["verdict"] != "supported_no_material_regression")
        ):
            raise ValueError(
                "strict comparator changed the fixed policy or work contract"
            )
        reports.append(
            {
                "configuration": config["name"],
                "output": str(target),
                "sha256": sha(target),
                "verdict": report["verdict"],
                "argv": command,
            }
        )
        manifest["comparisons"] = reports
        checkpoint(output / "manifest.json", manifest)
    return reports


def interrupt_campaign(signum, _frame):
    """Route SIGTERM through the same cleanup path as a keyboard interrupt."""
    # Let the first interruption finish child cleanup and evidence writes.
    signal.signal(signum, signal.SIG_IGN)
    raise KeyboardInterrupt(
        f"campaign interrupted by {signal.Signals(signum).name}"
    )


def execute(plan, manifest, output):
    """Scope SIGTERM handling to campaign execution and restore the caller."""
    previous_sigterm = signal.signal(signal.SIGTERM, interrupt_campaign)
    try:
        return execute_campaign(plan, manifest, output)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def execute_campaign(plan, manifest, output):
    """Execute a pristine campaign once, checkpointing any partial failure."""
    verify_frozen(plan)
    write_new(
        output / "execution-claim.json", {"utc": now(), "pid": os.getpid()}
    )
    (output / "raw").mkdir()
    (output / "reports").mkdir()
    manifest.update(status="running", started_utc=now())
    checkpoint(output / "manifest.json", manifest)
    configs = {row["name"]: row for row in plan["source"]["configurations"]}
    signatures, fixture_hashes, work, window = {}, {}, {}, 1
    try:
        for original in plan["runs"]:
            if original["window"] != window:
                manifest.setdefault("window_pauses", []).append(
                    {
                        "started_utc": now(),
                        "seconds": plan["window_pause_seconds"],
                    }
                )
                checkpoint(output / "manifest.json", manifest)
                time.sleep(plan["window_pause_seconds"])
                manifest["window_pauses"][-1]["completed_utc"] = now()
                window = original["window"]
            run = {
                **copy.deepcopy(original),
                "status": "running",
                "started_utc": now(),
            }
            manifest["runs"].append(run)
            checkpoint(output / "manifest.json", manifest)
            config = configs[run["configuration"]]
            stage = plan["source"]["stages"][run["side"]]
            if pathlib.Path(run["output"]).exists():
                raise ValueError("raw artifact already exists")
            run["telemetry_before"] = telemetry()
            checkpoint(output / "manifest.json", manifest)
            start = time.monotonic_ns()
            try:
                returncode = run_command(
                    run["argv"],
                    cwd=output,
                    env=environment(stage, config["spec"]),
                    log=run["log"],
                    timeout=plan["worker_timeout_seconds"],
                )
                run["returncode"] = returncode
            finally:
                run["child_wall_ns"] = time.monotonic_ns() - start
                run["telemetry_after"] = telemetry()
            if returncode:
                raise RuntimeError(f"benchmark child exited {returncode}")
            data = json.loads(pathlib.Path(run["output"]).read_text())
            signature = audit_raw(data, config, stage)
            for case in data["cases"]:
                fixture = json.dumps(case["case"], sort_keys=True)
                digest = case["input_sha256"]
                if (
                    fixture in fixture_hashes
                    and fixture_hashes[fixture] != digest
                ):
                    raise ValueError(
                        "fixture bytes changed across configurations/processes"
                    )
                fixture_hashes[fixture] = digest
                for entry in case["results"]:
                    key = (run["configuration"], fixture, entry["name"])
                    counts = tuple(
                        entry[name]
                        for name in ("iterations", "matvecs", "restarts")
                    )
                    if key in work and work[key] != counts:
                        raise ValueError(
                            "same-candidate numerical work changed across processes/sides"
                        )
                    work[key] = counts
            name = run["configuration"]
            if name in signatures and signatures[name] != signature:
                raise ValueError(
                    "semantic environment changed within configuration"
                )
            signatures[name] = signature
            run.update(
                status="measured",
                completed_utc=now(),
                sha256=sha(run["output"]),
                log_sha256=sha(run["log"]),
            )
            checkpoint(output / "manifest.json", manifest)
            print(
                f"DONE {len(manifest['runs'])}/{len(plan['runs'])} {run['id']}",
                flush=True,
            )
        verify_frozen(plan)
        for run in manifest["runs"]:
            if (
                sha(run["output"]) != run["sha256"]
                or sha(run["log"]) != run["log_sha256"]
            ):
                raise ValueError("completed evidence changed")
        manifest["comparisons"] = compare_all(plan, manifest, output)
        verify_frozen(plan)
        verdicts = {row["verdict"] for row in manifest["comparisons"]}
        verdict = (
            "regression"
            if "regression" in verdicts
            else "inconclusive"
            if "inconclusive" in verdicts
            else "supported_no_material_regression"
        )
        manifest.update(
            status="complete",
            completed_utc=now(),
            verdict=verdict,
            environments=signatures,
        )
        checkpoint(output / "manifest.json", manifest)
        return int(verdict != "supported_no_material_regression")
    except BaseException as error:
        if manifest["runs"] and manifest["runs"][-1]["status"] == "running":
            run = manifest["runs"][-1]
            run.update(
                status="failed", failure=f"{type(error).__name__}: {error}"
            )
            run["retained_files"] = {
                key: {"path": run[key], "sha256": sha(run[key])}
                for key in ("output", "log")
                if pathlib.Path(run[key]).is_file()
            }
        manifest.update(
            status="failed",
            failed_utc=now(),
            failure=f"{type(error).__name__}: {error}",
        )
        checkpoint(output / "manifest.json", manifest)
        print(
            f"STOP: {manifest['failure']}; all partial evidence retained",
            file=sys.stderr,
            flush=True,
        )
        if not isinstance(error, Exception):
            raise
        return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        input_path, output = (
            args.plan.resolve(strict=True),
            args.output_dir.resolve(),
        )
        if input_path.is_relative_to(output):
            raise ValueError(
                "input plan must be outside the immutable output directory"
            )
        plan = make_plan(input_path, output)
        manifest = prepare_output(plan, output, args.plan_only)
        if args.plan_only:
            print(f"PLANNED {len(plan['runs'])} fresh-process slots: {output}")
            return 0
        return execute(plan, manifest, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
