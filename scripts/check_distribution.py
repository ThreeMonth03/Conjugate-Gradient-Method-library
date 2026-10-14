"""Build an sdist in PEP 517 isolation and test its relocated installed wheel."""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
from pathlib import Path


def check_sdist(path):
    """Require build sources and reject local/generated/research artifacts."""
    with tarfile.open(path) as archive:
        members = archive.getmembers()
    names = set()
    forbidden = {".venv", ".deps", "_deps", ".cache", "build", "dist", "__pycache__", ".git"}
    for member in members:
        parts = Path(member.name).parts
        if not parts or Path(member.name).is_absolute() or ".." in parts:
            raise AssertionError(f"unsafe sdist path: {member.name}")
        if forbidden.intersection(parts):
            raise AssertionError(f"local build/dependency included: {member.name}")
        if member.issym() or member.islnk():
            raise AssertionError(f"unexpected sdist symlink: {member.name}")
        names.add("/".join(parts[1:]))
    required = {
        "CMakeLists.txt",
        "CMakePresets.json",
        "pyproject.toml",
        ".clang-format",
        ".clang-tidy",
        "cmake/CGOpenBLAS.cmake",
        "cmake/NamespaceOpenBLAS.cmake",
        "src/blas.cpp",
        "include/cg/detail/blas.hpp",
        "python/cgpy/__init__.py",
        "python/cgbench/cli.py",
        "docs/conf.py",
        "scripts/check_installed.py",
        "tests/cpp/test_blas.cpp",
    }
    if not required <= names:
        raise AssertionError(f"sdist missing sources: {sorted(required - names)}")
    research = {"benchmarks/compare_direct.py", "benchmarks/cg_vs_direct_results.json"}
    if names.intersection(research) or any(n.startswith("benchmarks/results/") for n in names):
        raise AssertionError("research or raw measurement artifacts leaked into sdist")
    print(f"Checked {len(names)} sdist paths; no local dependencies or research files.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, required=True)
    parser.add_argument("--blas", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--jobs", type=int, default=2)
    options = parser.parse_args()
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    root = Path(__file__).resolve().parents[1]
    sdist = options.sdist.resolve()
    check_sdist(sdist)
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.update(
        PIP_CACHE_DIR=str(root / ".cache/pip"),
        CMAKE_BUILD_PARALLEL_LEVEL=str(options.jobs),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    # Source archive, wheel and installation all move outside the repository.
    # Only the download cache remains project-local; the temporary venv is
    # removed on success or failure and never installs into the host Python.
    with tempfile.TemporaryDirectory(prefix="cgpy-distribution-") as temporary:
        work = Path(temporary)
        moved_sdist = work / sdist.name
        shutil.copyfile(sdist, moved_sdist)
        wheels = work / "wheels"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-cache-dir",
                "--wheel-dir",
                str(wheels),
                "--config-settings",
                f"cmake.define.CG_ENABLE_BLAS={options.blas}",
                "--config-settings",
                f"cmake.define.CG_BLAS_BUILD_JOBS={options.jobs}",
                str(moved_sdist),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        wheel_files = list(wheels.glob("cgpy-*.whl"))
        if len(wheel_files) != 1:
            raise AssertionError(f"expected one wheel, found {wheel_files}")
        installation = work / "installed"
        venv.EnvBuilder(with_pip=True).create(installation)
        python = installation / "bin/python"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--constraint",
                str(root / "requirements/packaging.lock"),
                str(wheel_files[0]),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        check = work / "check_installed.py"
        shutil.copyfile(root / "scripts/check_installed.py", check)
        subprocess.run(
            [str(python), "-I", str(check), "--blas", options.blas],
            cwd=work,
            env=environment,
            check=True,
        )
    print(f"Relocated sdist → isolated wheel → installed smoke passed (BLAS={options.blas}).")


if __name__ == "__main__":
    main()
