"""Exercise campaign provenance, fixed schedules and failure retention without timing."""

import copy
import json
import pathlib
import signal
import subprocess

import pytest

from scripts import run_performance_campaign as campaign


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf8")


@pytest.fixture
def declared(tmp_path):
    root = tmp_path / "stage" / "python"
    files = {}
    for name in (
        "cgbench/__main__.py",
        "cgpy/__init__.py",
        "cgpy/_core.fake.so",
    ):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("frozen payload", encoding="utf8")
        files[str(path)] = campaign.sha(path)
    stage = {
        "package_root": str(root),
        "compiled_revision": "a" * 40,
        "compiled_tree": "b" * 40,
        "files": files,
    }
    python = tmp_path / "python"
    python.write_text("frozen interpreter", encoding="utf8")
    comparator = tmp_path / "compare.py"
    comparator.write_text("frozen comparator", encoding="utf8")
    spec = {
        "comparison": "implementations",
        "suite": "smoke",
        "scope": "prepared",
        "cases": [{"name": "dense-4", "size": 4, "condition_scale": 100.0}],
        "implementations": ["native", "numpy-out", "torch-eager-out"],
        "threads": 1,
        "seed": 17,
        "solver": {
            "algorithm": "standard",
            "rtol": 1e-8,
            "atol": 1e-12,
            "maxiter": 100,
            "residual_refresh": 50,
            "fused": True,
            "preconditioner": "none",
        },
        "sampling": {"warmups": 3, "rounds": 6, "repetitions": 1},
    }
    configs = [
        {
            "name": name,
            "python": str(python),
            "preset": "cpu-release",
            "spec": copy.deepcopy(spec),
            "affinity": [0],
        }
        for name in ("first", "second")
    ]
    source = {
        "schema_version": 1,
        "mode": "calibration",
        "comparator": str(comparator),
        "stages": {"baseline": stage, "current": copy.deepcopy(stage)},
        "configurations": configs,
        "window_pause_seconds": 0,
        "worker_timeout_seconds": 10,
    }
    input_path, output = tmp_path / "input.json", tmp_path / "results"
    save(input_path, source)
    return source, input_path, output


def planned(declared):
    source, input_path, output = declared
    save(input_path, source)
    plan = campaign.make_plan(input_path, output)
    manifest = campaign.prepare_output(plan, output, True)
    return plan, manifest, output


def raw_artifact(config, stage):
    spec = copy.deepcopy(config["spec"])
    names = spec["implementations"]
    # The six permutations are the complete three-candidate Williams cycle.
    orders = [
        [names[index] for index in row]
        for row in (
            (0, 1, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
            (0, 2, 1),
            (1, 0, 2),
        )
    ]
    results = [
        {
            "name": name,
            "status": "measured",
            "solver_status": "converged",
            "iterations": 4,
            "matvecs": 6,
            "restarts": 0,
            "true_residual_norm": 1e-10,
            "residual_threshold": 1e-8,
            "round_elapsed_ns": [100] * 6,
            "setup": {
                "preconditioner": spec["solver"]["preconditioner"],
                "setup_in_timed_call": spec["scope"] == "api",
                "residual_policy": spec["solver"]["algorithm"]
                + " with residual_refresh="
                + str(spec["solver"]["residual_refresh"]),
                **({"dense_kernel": "native"} if name == "native" else {}),
            },
        }
        for name in names
    ]
    native = next(name for name in stage["files"] if name.endswith(".so"))
    env = {
        "affinity": config["affinity"],
        "python_executable": config["python"],
        "thread_environment": {name: "1" for name in campaign.THREADS},
        "dynamic_thread_environment": {
            "OMP_DYNAMIC": "FALSE",
            "MKL_DYNAMIC": "FALSE",
        },
        "threadpools": [{"num_threads": 1, "prefix": "openblas"}],
        "torch": {"intraop_threads": 1, "interop_threads": 1},
        "native": {
            "build_info": {"openmp": False, "blas": {"enabled": False}},
            "python_package_path": str(
                pathlib.Path(stage["package_root"]) / "cgpy/__init__.py"
            ),
            "module_path": native,
            "module_sha256": stage["files"][native],
        },
        "cmake": None,
    }
    return {
        "schema_version": 1,
        "metric": "time_to_solution",
        "spec": spec,
        "environment": env,
        "cases": [
            {
                "case": spec["cases"][0],
                "dtype": "float64",
                "sampling": spec["sampling"],
                "schedule": "williams",
                "cycle_rows": 6,
                "round_orders": orders,
                "input_sha256": "c" * 64,
                "results": results,
            }
        ],
    }


def report_for(command, config, verdict):
    baseline = command[
        command.index("--baseline") + 1 : command.index("--current")
    ]
    current = command[
        command.index("--current") + 1 : command.index("--max-regression")
    ]
    return {
        "max_regression": 0.05,
        "ratio_limit": 1.05,
        "baseline_files": baseline,
        "current_files": current,
        "bootstrap": {
            "resamples": 10000,
            "seed": 20261018,
            "confidence": 0.95,
        },
        "restart_parity_verified": True,
        "verdict": verdict,
        "benchmark": {
            "metric": "time_to_solution",
            "spec": {
                key: value
                for key, value in config["spec"].items()
                if key != "sampling"
            },
        },
        "cases": [
            {"case": case, "implementation": name}
            for case in config["spec"]["cases"]
            for name in config["spec"]["implementations"]
        ],
    }


def install_children(
    monkeypatch,
    plan,
    *,
    verdict="supported_no_material_regression",
    mutate=None,
):
    calls = []
    configs = {row["name"]: row for row in plan["source"]["configurations"]}
    runs = {tuple(row["argv"]): row for row in plan["runs"]}

    def child(command, *, cwd, env, log, timeout):
        calls.append(command)
        pathlib.Path(log).write_text("retained child output", encoding="utf8")
        target = pathlib.Path(command[command.index("--output") + 1])
        if tuple(command) in runs:
            run = runs[tuple(command)]
            config = configs[run["configuration"]]
            value = raw_artifact(config, plan["source"]["stages"][run["side"]])
            if mutate:
                mutate(run, value)
            save(target, value)
            return 0
        config = configs[target.stem]
        save(target, report_for(command, config, verdict))
        return int(verdict != "supported_no_material_regression")

    monkeypatch.setattr(campaign, "run_command", child)
    monkeypatch.setattr(
        campaign,
        "telemetry",
        lambda: {"utc": "diagnostic boundary", "monotonic_ns": 1},
    )
    monkeypatch.setattr(campaign.time, "sleep", lambda seconds: None)
    return calls


def test_schedule_balances_every_slot_and_reverses_configuration_order(
    declared,
):
    plan, _manifest, _output = planned(declared)
    assert len(plan["runs"]) == 32
    assert plan["policy"]["max_regression"] == 0.05
    for window in (1, 2):
        selected = [row for row in plan["runs"] if row["window"] == window]
        assert list(
            dict.fromkeys(row["configuration"] for row in selected)
        ) == (["first", "second"] if window == 1 else ["second", "first"])
        for name in ("first", "second"):
            runs = [row for row in selected if row["configuration"] == name]
            assert [row["side"] for row in runs] == [
                *campaign.QUARTETS[0],
                *campaign.QUARTETS[1],
            ]
            for slot in range(1, 5):
                assert sorted(
                    row["side"] for row in runs if row["slot"] == slot
                ) == ["baseline", "current"]
    assert len({row["output"] for row in plan["runs"]}) == 32
    for row in plan["runs"]:
        assert row["argv"][4:7] == ["-m", "cgbench", "run"]
        assert row["argv"][-2:] == ["--output", row["output"]]


@pytest.mark.parametrize(
    "tamper",
    ["payload", "control", "duplicate", "preset", "affinity", "comparison"],
)
def test_invalid_input_cannot_start_children(declared, tamper):
    source, input_path, output = declared
    if tamper == "payload":
        source["stages"]["baseline"]["files"].pop(
            next(iter(source["stages"]["baseline"]["files"]))
        )
    elif tamper == "control":
        source["configurations"][0]["spec"]["implementations"].remove(
            "numpy-out"
        )
    elif tamper == "duplicate":
        source["configurations"][1]["name"] = "first"
    elif tamper == "preset":
        source["configurations"][1]["preset"] = "cpu-blas"
    elif tamper == "affinity":
        source["configurations"][0]["affinity"] = [False]
    else:
        source["mode"] = "comparison"
    save(input_path, source)
    with pytest.raises(ValueError):
        campaign.make_plan(input_path, output)
    assert not output.exists()


def test_only_identical_pristine_plan_can_execute(declared):
    plan, manifest, output = planned(declared)
    assert campaign.prepare_output(plan, output, False) == manifest
    with pytest.raises(ValueError, match="pristine"):
        campaign.prepare_output(plan, output, True)
    changed = copy.deepcopy(plan)
    changed["runs"][0]["side"] = "current"
    with pytest.raises(ValueError, match="pristine"):
        campaign.prepare_output(changed, output, False)
    manifest["runs"].append({"status": "started"})
    save(output / "manifest.json", manifest)
    with pytest.raises(ValueError, match="pristine"):
        campaign.prepare_output(plan, output, False)


@pytest.mark.parametrize(
    "verdict",
    ["supported_no_material_regression", "inconclusive", "regression"],
)
def test_complete_campaign_retains_all_controls_and_primary_verdict(
    declared, monkeypatch, verdict
):
    plan, manifest, output = planned(declared)
    calls = install_children(monkeypatch, plan, verdict=verdict)
    assert campaign.execute(plan, manifest, output) == int(
        verdict != "supported_no_material_regression"
    )
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "complete"
    assert saved["verdict"] == verdict
    assert len(saved["runs"]) == 32
    assert len(calls) == 34
    assert len(saved["comparisons"]) == 2
    for row in saved["runs"]:
        assert row["status"] == "measured"
        assert row["sha256"] == campaign.sha(row["output"])
        assert "telemetry_before" in row and "telemetry_after" in row
    for comparison in saved["comparisons"]:
        report = json.loads(pathlib.Path(comparison["output"]).read_text())
        assert (
            len(report["baseline_files"]) == len(report["current_files"]) == 8
        )
        assert {row["implementation"] for row in report["cases"]} == {
            "native",
            "numpy-out",
            "torch-eager-out",
        }
    with pytest.raises(ValueError, match="pristine"):
        campaign.prepare_output(plan, output, False)


@pytest.mark.parametrize("failure", ["returncode", "timeout", "interrupt"])
def test_child_failures_retain_partial_evidence_without_retry(
    declared, monkeypatch, failure
):
    plan, manifest, output = planned(declared)
    calls = []

    def child(command, *, cwd, env, log, timeout):
        calls.append(command)
        pathlib.Path(log).write_text("retained partial log", encoding="utf8")
        raw = pathlib.Path(command[command.index("--output") + 1])
        raw.write_text("retained partial artifact", encoding="utf8")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, timeout)
        if failure == "interrupt":
            raise KeyboardInterrupt()
        return 9

    monkeypatch.setattr(campaign, "run_command", child)
    monkeypatch.setattr(campaign, "telemetry", lambda: {"unavailable": True})
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            campaign.execute(plan, manifest, output)
    else:
        assert campaign.execute(plan, manifest, output) == 1
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == saved["runs"][0]["status"] == "failed"
    assert len(calls) == 1
    assert set(saved["runs"][0]["retained_files"]) == {"output", "log"}
    assert "telemetry_after" in saved["runs"][0]
    with pytest.raises(ValueError, match="pristine"):
        campaign.prepare_output(plan, output, False)


@pytest.mark.parametrize(
    "field",
    [
        "scope",
        "sampling",
        "control",
        "schedule",
        "counts",
        "residual",
        "hash",
        "environment",
    ],
)
def test_corrupt_artifacts_fail_immediately_and_remain_visible(
    declared, monkeypatch, field
):
    plan, manifest, output = planned(declared)

    def mutate(run, value):
        if field == "scope":
            value["spec"]["scope"] = "api"
        elif field == "sampling":
            value["cases"][0]["sampling"] = {"rounds": 1}
        elif field == "control":
            value["cases"][0]["results"].pop()
        elif field == "schedule":
            value["cases"][0]["round_orders"] = [
                value["cases"][0]["round_orders"][0]
            ] * 6
        elif field == "counts":
            value["cases"][0]["results"][0]["restarts"] = None
        elif field == "residual":
            value["cases"][0]["results"][0]["true_residual_norm"] = 1
        elif field == "hash":
            value["cases"][0]["input_sha256"] = "invalid"
        else:
            value["environment"]["threadpools"][0]["num_threads"] = 8

    calls = install_children(monkeypatch, plan, mutate=mutate)
    assert campaign.execute(plan, manifest, output) == 1
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["status"] == "failed"
    assert len(calls) == 1
    assert pathlib.Path(saved["runs"][0]["output"]).exists()


def test_changed_known_work_fails_before_any_normalization(
    declared, monkeypatch
):
    plan, manifest, output = planned(declared)

    def mutate(run, value):
        if run["side"] == "current":
            value["cases"][0]["results"][0]["matvecs"] += 1

    calls = install_children(monkeypatch, plan, mutate=mutate)
    assert campaign.execute(plan, manifest, output) == 1
    assert len(calls) == 2
    assert "numerical work changed" in manifest["failure"]


@pytest.mark.parametrize("field", ["device", "cuda_device", "execution_mode"])
def test_noncanonical_public_specs_are_rejected_before_measurement(
    declared, field
):
    source, input_path, output = declared
    spec = source["configurations"][0]["spec"]
    if field == "execution_mode":
        spec["solver"][field] = "host"
    else:
        spec[field] = "cpu" if field == "device" else 0
    save(input_path, source)
    with pytest.raises(ValueError, match="canonical"):
        campaign.make_plan(input_path, output)
    assert not output.exists()


def cuda_setup(spec, name):
    mode = spec["solver"].get("execution_mode", "host")
    setup = {
        "preconditioner": spec["solver"]["preconditioner"],
        "setup_in_timed_call": spec["scope"] in ("api", "session"),
        "residual_policy": "standard with residual_refresh=50",
        "cuda_device": spec["cuda_device"],
        "stream": 0,
    }
    if name == "torch-cuda-eager":
        setup.update(
            native_execution_mode=mode, native_execution_mode_applies=False
        )
    else:
        setup.update(
            execution_mode=mode,
            fused=spec["solver"]["fused"],
            dense_kernel="cublas" if name == "cuda-cublas" else "native",
        )
    if spec["scope"] == "session":
        setup.update(
            workspace_reused=True,
            workspace_in_timed_call=False,
            output_reused=False,
            output_in_timed_call=True,
            preconditioner_reused=False,
            preconditioner_in_timed_call=spec["solver"]["preconditioner"]
            == "jacobi",
        )
    return {"name": name, "setup": setup}


@pytest.mark.parametrize("scope", ["api", "session", "prepared"])
@pytest.mark.parametrize(
    "name", ["cuda-native", "cuda-cublas", "torch-cuda-eager"]
)
@pytest.mark.parametrize("mode", ["host", "device"])
def test_cuda_adapter_scopes_and_torch_mode_disclosure(
    declared, scope, name, mode
):
    spec = copy.deepcopy(declared[0]["configurations"][0]["spec"])
    spec.update(device="cuda", cuda_device=0, scope=scope)
    spec["solver"].update(preconditioner="jacobi")
    if mode == "device":
        spec["solver"]["execution_mode"] = mode
    entry = cuda_setup(spec, name)
    campaign.audit_setup(entry, spec, {"blas": {"enabled": False}})
    entry["setup"]["setup_in_timed_call"] = not entry["setup"][
        "setup_in_timed_call"
    ]
    with pytest.raises(ValueError, match="timing scope"):
        campaign.audit_setup(entry, spec, {"blas": {"enabled": False}})


@pytest.mark.parametrize(
    "field",
    [
        "workspace_reused",
        "workspace_in_timed_call",
        "output_reused",
        "output_in_timed_call",
        "preconditioner_reused",
        "preconditioner_in_timed_call",
    ],
)
@pytest.mark.parametrize("name", ["cuda-cublas", "torch-cuda-eager"])
def test_session_cannot_hide_output_or_preconditioner_reuse(
    declared, field, name
):
    spec = copy.deepcopy(declared[0]["configurations"][0]["spec"])
    spec.update(device="cuda", cuda_device=0, scope="session")
    spec["solver"]["preconditioner"] = "jacobi"
    entry = cuda_setup(spec, name)
    entry["setup"][field] = not entry["setup"][field]
    with pytest.raises(ValueError, match="Session"):
        campaign.audit_setup(entry, spec, {"blas": {"enabled": False}})


@pytest.mark.parametrize(
    "field",
    [
        "preconditioner",
        "residual_policy",
        "execution_mode",
        "dense_kernel",
        "fused",
        "stream",
        "cuda_device",
    ],
)
def test_cuda_adapter_cannot_mislabel_backend_or_algorithm(declared, field):
    spec = copy.deepcopy(declared[0]["configurations"][0]["spec"])
    spec.update(device="cuda", cuda_device=0)
    entry = cuda_setup(spec, "cuda-cublas")
    entry["setup"][field] = "incorrect"
    with pytest.raises(ValueError):
        campaign.audit_setup(entry, spec, {"blas": {"enabled": False}})


@pytest.mark.parametrize(
    "field", ["native_execution_mode", "native_execution_mode_applies"]
)
def test_torch_reference_cannot_claim_to_execute_native_mode(declared, field):
    spec = copy.deepcopy(declared[0]["configurations"][0]["spec"])
    spec.update(device="cuda", cuda_device=0)
    spec["solver"]["execution_mode"] = "device"
    entry = cuda_setup(spec, "torch-cuda-eager")
    entry["setup"][field] = (
        "host" if field == "native_execution_mode" else True
    )
    with pytest.raises(ValueError, match="Torch"):
        campaign.audit_setup(entry, spec, {"blas": {"enabled": False}})


def test_cpu_blas_adapter_must_report_the_actual_provider(declared):
    spec = declared[0]["configurations"][0]["spec"]
    entry = {
        "name": "native-blas",
        "setup": {
            "preconditioner": "none",
            "setup_in_timed_call": False,
            "residual_policy": "standard with residual_refresh=50",
            "dense_kernel": "blas",
            "blas": {"enabled": True, "provider": "measured"},
        },
    }
    build = {"blas": {"enabled": True, "provider": "measured"}}
    campaign.audit_setup(entry, spec, build)
    entry["setup"]["blas"] = {"enabled": True, "provider": "another"}
    with pytest.raises(ValueError, match="provider"):
        campaign.audit_setup(entry, spec, build)


def test_cuda_environment_is_explicit_without_host_mutation(monkeypatch):
    monkeypatch.setenv("CUDA_LAUNCH_BLOCKING", "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    monkeypatch.setenv("PYTHONHOME", "/unrelated/python")
    monkeypatch.setenv("PYTHONSTARTUP", "/unrelated/startup.py")
    env = campaign.environment(
        {"package_root": "/local/stage"}, {"device": "cuda", "threads": 1}
    )
    assert env["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert "CUDA_LAUNCH_BLOCKING" not in env
    assert "PYTHONHOME" not in env and "PYTHONSTARTUP" not in env
    assert all(env[name] == "1" for name in campaign.THREADS)
    assert campaign.os.environ["CUDA_LAUNCH_BLOCKING"] == "1"


def test_missing_telemetry_is_explicit_and_never_synthesizes_zero(monkeypatch):
    def missing(_path, *args, **kwargs):
        raise OSError("sensor unavailable")

    monkeypatch.setattr(pathlib.Path, "read_text", missing)
    monkeypatch.setattr(pathlib.Path, "glob", lambda *args: [])
    monkeypatch.setattr(pathlib.Path, "is_file", lambda path: False)
    monkeypatch.setattr(campaign.shutil, "which", lambda name: None)
    snapshot = campaign.telemetry()
    assert (
        snapshot["cpu_frequency"]
        == snapshot["cpu_temperature"]
        == "unavailable"
    )
    assert all(
        row["value"] is None and row["error"]
        for row in snapshot["files"].values()
    )
    assert snapshot["cpu_mhz"]["value"] is None
    assert snapshot["gpu"]["returncode"] is None
    assert snapshot["gpu"]["error"] == "nvidia-smi unavailable"


def test_timeout_terminates_the_launcher_and_worker_group(
    tmp_path, monkeypatch
):
    calls = []

    class Process:
        pid = 12345

        def wait(self, timeout=None):
            calls.append(("wait", timeout))
            if len([row for row in calls if row[0] == "wait"]) < 3:
                raise subprocess.TimeoutExpired(["fake"], timeout)
            return -9

    def launch(command, **kwargs):
        assert kwargs["start_new_session"] is True
        return Process()

    monkeypatch.setattr(campaign.subprocess, "Popen", launch)
    monkeypatch.setattr(
        campaign.os, "killpg", lambda pid, signum: calls.append((pid, signum))
    )
    with pytest.raises(subprocess.TimeoutExpired):
        campaign.run_command(
            ["fake"],
            cwd=tmp_path,
            env={},
            log=tmp_path / "child.log",
            timeout=1,
        )
    assert (12345, signal.SIGTERM) in calls
    assert (12345, signal.SIGKILL) in calls
