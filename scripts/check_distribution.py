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
    forbidden = {
        ".venv",
        ".venv-cuda",
        ".deps",
        "_deps",
        ".cache",
        "build",
        "dist",
        "__pycache__",
        ".git",
    }
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
        "src/cuda_storage.cu",
        "src/cuda_backend.cu",
        "src/cuda_solver.cu",
        "include/cg/detail/blas.hpp",
        "include/cg/detail/solver_engine.hpp",
        "include/cg/cuda.hpp",
        "include/cg/detail/cuda.hpp",
        "bindings/cuda.cpp",
        "python/cgpy/__init__.py",
        "python/cgpy/cuda.py",
        "python/cgbench/cli.py",
        "docs/conf.py",
        "scripts/check_installed.py",
        "scripts/setup_cuda.py",
        "requirements/cuda.lock",
        "requirements/cuda-benchmark.lock",
        "tests/cpp/test_blas.cpp",
        "tests/cpp/test_cuda.cpp",
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
    parser.add_argument("--cuda", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--cuda-root", type=Path)
    parser.add_argument("--cuda-host-compiler", default="/usr/bin/g++-13")
    parser.add_argument("--cuda-architectures", default="75")
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
    environment.pop("LD_LIBRARY_PATH", None)
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
        build_command = [
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
            "--config-settings",
            f"cmake.define.CG_ENABLE_CUDA={options.cuda}",
        ]
        if options.cuda == "ON":
            toolkit = (options.cuda_root or root / ".deps/cuda-12.8").resolve()
            build_command.extend(
                [
                    "--config-settings",
                    f"cmake.define.CUDAToolkit_ROOT={toolkit}",
                    "--config-settings",
                    f"cmake.define.CMAKE_CUDA_COMPILER={toolkit / 'bin/nvcc'}",
                    "--config-settings",
                    f"cmake.define.CMAKE_CUDA_HOST_COMPILER={options.cuda_host_compiler}",
                    "--config-settings",
                    f"cmake.define.CMAKE_CXX_COMPILER={options.cuda_host_compiler}",
                    "--config-settings",
                    f"cmake.define.CMAKE_CUDA_ARCHITECTURES={options.cuda_architectures}",
                ]
            )
        build_command.append(str(moved_sdist))
        subprocess.run(
            build_command,
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
                str(wheel_files[0]) + ("[cuda]" if options.cuda == "ON" else ""),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        check = work / "check_installed.py"
        shutil.copyfile(root / "scripts/check_installed.py", check)
        subprocess.run(
            [str(python), "-I", str(check), "--blas", options.blas, "--cuda", options.cuda],
            cwd=work,
            env=environment,
            check=True,
        )
    print(
        f"Relocated sdist → isolated wheel → installed smoke passed (BLAS={options.blas}, CUDA={options.cuda})."
    )


if __name__ == "__main__":
    main()
