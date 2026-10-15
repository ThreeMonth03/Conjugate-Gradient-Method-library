"""Build an sdist in PEP 517 isolation and test its relocated installed wheel."""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv


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
        parts = pathlib.Path(member.name).parts
        if (
            not parts
            or pathlib.Path(member.name).is_absolute()
            or ".." in parts
        ):
            raise AssertionError(f"unsafe sdist path: {member.name}")
        if forbidden.intersection(parts):
            raise AssertionError(
                f"local build/dependency included: {member.name}"
            )
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
        "cmake/CGCuda.cmake",
        "cmake/CGPython.cmake",
        "cmake/CGCudaToolchain.cmake",
        "cpp/cg/backend/cpu/blas.cpp",
        "cpp/cg/backend/cuda/backend.cu",
        "cpp/cg/backend/cuda/detail/reduction.cuh",
        "cpp/cg/storage/cuda/allocation.cu",
        "cpp/cg/storage/cuda/device_memory.hpp",
        "cpp/cg/storage/cuda/vector.cu",
        "cpp/cg/matrix/cuda/csr_matrix.cu",
        "cpp/cg/solver/cuda/conjugate_gradient.cu",
        "cpp/cg/solver/detail/engine.hpp",
        "cpp/cg/cuda.hpp",
        "cpp/cg/backend/cuda/detail/runtime.hpp",
        "cpp/cg/solver/pymod/cuda.cpp",
        "python/cgpy/__init__.py",
        "python/cgpy/cuda/__init__.py",
        "python/cgbench/cli.py",
        "docs/conf.py",
        "docs/style.rst",
        "docs/_static/docs.css",
        "docs/benchmarks.rst",
        "docs/benchmarks/methodology.rst",
        "docs/benchmarks/cpu.rst",
        "docs/benchmarks/sparse.rst",
        "docs/benchmarks/cuda.rst",
        "scripts/run_paired_performance.py",
        "scripts/check_installed.py",
        "scripts/setup_cuda.py",
        "requirements/cuda.lock",
        "requirements/cuda-benchmark.lock",
        "tests/cpp/backend/test_blas.cpp",
        "tests/cpp/cuda/test_cuda.cpp",
    }
    if not required <= names:
        raise AssertionError(
            f"sdist missing sources: {sorted(required - names)}"
        )
    retired = {
        "STYLE.md",
        "docs/refactor_validation.rst",
        "cpp/cg/storage/vector.hpp",
        "cpp/cg/storage/vector.cpp",
        "python/utils.py",
        "cpp/matrix/_matrix.cpp",
        "cpp/pybind/_pybind.cpp",
    }
    if names.intersection(retired):
        raise AssertionError("retired APIs leaked into sdist")
    if any(n.startswith("benchmarks/") for n in names):
        raise AssertionError(
            "research or raw measurement artifacts leaked into sdist"
        )
    print(
        f"Checked {len(names)} sdist paths; no local dependencies or research files.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=pathlib.Path, required=True)
    parser.add_argument("--blas", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--cuda", choices=("OFF", "ON"), default="OFF")
    parser.add_argument("--cuda-root", type=pathlib.Path)
    parser.add_argument("--cuda-host-compiler")
    parser.add_argument("--cuda-architectures")
    parser.add_argument("--jobs", type=int, default=2)
    options = parser.parse_args()
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    root = pathlib.Path(__file__).resolve().parents[1]
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
        work = pathlib.Path(temporary)
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
                    "cmake.define.CMAKE_TOOLCHAIN_FILE=cmake/CGCudaToolchain.cmake",
                    "--config-settings",
                    f"cmake.define.CUDAToolkit_ROOT={toolkit}",
                ]
            )
            for name, value in (
                ("CMAKE_CUDA_HOST_COMPILER", options.cuda_host_compiler),
                ("CMAKE_CUDA_ARCHITECTURES", options.cuda_architectures),
            ):
                if value:
                    build_command.extend(
                        ["--config-settings", f"cmake.define.{name}={value}"]
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
                str(wheel_files[0])
                + ("[cuda]" if options.cuda == "ON" else ""),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        check = work / "check_installed.py"
        shutil.copyfile(root / "scripts/check_installed.py", check)
        subprocess.run(
            [
                str(python),
                "-I",
                str(check),
                "--blas",
                options.blas,
                "--cuda",
                options.cuda,
            ],
            cwd=work,
            env=environment,
            check=True,
        )
    print(
        f"Relocated sdist -> isolated wheel -> installed smoke passed (BLAS={options.blas}, CUDA={options.cuda})."
    )


if __name__ == "__main__":
    main()
