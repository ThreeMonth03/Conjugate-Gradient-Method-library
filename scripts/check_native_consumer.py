"""Install, relocate and test the standalone C++ package consumer."""

import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import tempfile


def read_cache(path):
    values = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith(("//", "#")):
            continue
        key, separator, value = line.partition("=")
        if separator:
            values[key.partition(":")[0]] = value
    return values


def check_package_paths(prefix, source, build):
    files = list(prefix.rglob("*.cmake"))
    if not files:
        raise AssertionError("development installation has no CMake package")
    forbidden = {str(source), str(build)}
    for path in files:
        content = path.read_text()
        for value in forbidden:
            if value in content:
                raise AssertionError(
                    f"non-relocatable path in {path}: {value}"
                )


def check_consumer_cache(build, prefix, source, producer):
    cache = read_cache(build / "CMakeCache.txt")
    package = pathlib.Path(cache["cgpy_DIR"]).resolve()
    if not package.is_relative_to(prefix):
        raise AssertionError(f"consumer found the wrong package: {package}")
    if any(name.startswith(("Python_", "Python3_")) for name in cache):
        raise AssertionError("native consumer unexpectedly discovered Python")
    content = (build / "compile_commands.json").read_text()
    for path in (source / "cpp", producer):
        if str(path) in content:
            raise AssertionError(f"consumer uses private headers: {path}")
    explicit = set(
        shlex.split(
            cache.get("CMAKE_CXX_FLAGS", "")
            + " "
            + cache.get(
                f"CMAKE_CXX_FLAGS_{cache.get('CMAKE_BUILD_TYPE', '').upper()}",
                "",
            )
        )
    )
    openmp = set(shlex.split(cache.get("OpenMP_CXX_FLAGS", "")))
    openmp.difference_update(("-Xclang", "-Xpreprocessor"))
    for command in json.loads(content):
        arguments = command.get("arguments") or shlex.split(command["command"])
        inherited = set(arguments) - explicit
        if openmp & inherited:
            raise AssertionError("consumer inherits OpenMP compiler flags")
        if any(
            value.startswith(("-DCG_ENABLE_BLAS", "-D_OPENMP"))
            for value in inherited
        ):
            raise AssertionError("consumer inherits implementation macros")
        for index, value in enumerate(arguments):
            directory = None
            for flag in ("-I", "-isystem", "-iquote", "-idirafter"):
                if value == flag:
                    directory = arguments[index + 1]
                    break
                if value.startswith(flag):
                    directory = value[len(flag) :]
                    break
            if directory is None or directory in explicit:
                continue
            if value != flag and value in explicit:
                continue
            include = pathlib.Path(command["directory"]) / directory
            if any(
                (include / header).is_file()
                for header in ("cuda_runtime_api.h", "cublas_v2.h")
            ):
                raise AssertionError("consumer inherits CUDA SDK headers")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--cuda-root", type=pathlib.Path)
    parser.add_argument("--require-cuda", action="store_true")
    options = parser.parse_args()
    if options.jobs < 1:
        parser.error("--jobs must be positive")
    source = pathlib.Path(__file__).resolve().parents[1]
    producer = options.build_dir.resolve()
    cache_file = producer / "CMakeCache.txt"
    if not cache_file.is_file():
        parser.error(f"missing configured build: {cache_file}")
    cache = read_cache(cache_file)
    cuda = cache.get("CG_ENABLE_CUDA", "OFF").upper() in (
        "1",
        "ON",
        "TRUE",
        "YES",
        "Y",
    )
    if options.require_cuda and not cuda:
        parser.error("--require-cuda requires a CUDA-enabled producer build")
    for name in ("CMAKE_INSTALL_LIBDIR", "CMAKE_INSTALL_INCLUDEDIR"):
        if pathlib.Path(cache.get(name, "")).is_absolute():
            parser.error(f"relocation requires a relative {name}")

    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "CMAKE_PREFIX_PATH",
        "cgpy_DIR",
        "cgpy_ROOT",
    ):
        environment.pop(name, None)
    cmake = str(
        pathlib.Path(shutil.which(options.cmake) or options.cmake).resolve()
    )
    ctest = str(pathlib.Path(cmake).with_name("ctest"))
    configuration = cache.get("CMAKE_BUILD_TYPE", "Release") or "Release"
    settings = [
        "-G",
        cache["CMAKE_GENERATOR"],
        f"-DCMAKE_BUILD_TYPE={configuration}",
        "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
        "-DCMAKE_FIND_USE_PACKAGE_REGISTRY=OFF",
        "-DCMAKE_FIND_USE_SYSTEM_PACKAGE_REGISTRY=OFF",
    ]
    for name in (
        "CMAKE_CXX_COMPILER",
        "CMAKE_MAKE_PROGRAM",
        "CMAKE_CXX_FLAGS",
        f"CMAKE_CXX_FLAGS_{configuration.upper()}",
        "OpenMP_CXX_FLAGS",
        "OpenMP_CXX_LIB_NAMES",
    ):
        if cache.get(name):
            settings.append(f"-D{name}={cache[name]}")
    for name, value in cache.items():
        if name.startswith("OpenMP_") and name.endswith("_LIBRARY"):
            settings.append(f"-D{name}={value}")

    with tempfile.TemporaryDirectory(prefix="cgpy-native-") as temporary:
        work = pathlib.Path(temporary).resolve()
        installed = work / "original-prefix"
        subprocess.run(
            [
                cmake,
                "--install",
                str(producer),
                "--config",
                configuration,
                "--component",
                "development",
                "--prefix",
                str(installed),
            ],
            cwd=work,
            env=environment,
            check=True,
        )
        prefix = work / "relocated-prefix"
        installed.rename(prefix)
        check_package_paths(prefix, source, producer)
        settings.append(f"-DCMAKE_PREFIX_PATH={prefix}")
        consumer = work / "consumer"
        shutil.copytree(source / "examples/cpp", consumer)
        for mode in ("cpu", "cuda") if cuda else ("cpu",):
            build = work / f"build-{mode}"
            arguments = ["-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON"]
            if mode == "cuda":
                arguments = [
                    "-DCG_CONSUMER_CUDA=ON",
                    f"-DCG_CONSUMER_REQUIRE_CUDA={'ON' if options.require_cuda else 'OFF'}",
                ]
                toolkit = options.cuda_root or cache.get("CUDAToolkit_ROOT")
                if toolkit:
                    arguments.append(
                        f"-DCUDAToolkit_ROOT={pathlib.Path(toolkit).resolve()}"
                    )
            subprocess.run(
                [
                    cmake,
                    "-S",
                    str(consumer),
                    "-B",
                    str(build),
                    *settings,
                    *arguments,
                ],
                cwd=work,
                env=environment,
                check=True,
            )
            check_consumer_cache(build, prefix, source, producer)
            subprocess.run(
                [
                    cmake,
                    "--build",
                    str(build),
                    "--config",
                    configuration,
                    "--parallel",
                    str(options.jobs),
                ],
                cwd=work,
                env=environment,
                check=True,
            )
            subprocess.run(
                [
                    ctest,
                    "--test-dir",
                    str(build),
                    "--build-config",
                    configuration,
                    "--output-on-failure",
                    "--no-tests=error",
                ],
                cwd=work,
                env=environment,
                check=True,
            )

        cases = [
            ("unknown", "COMPONENTS Unknown", "Unknown cgpy component", False),
            (
                "mixed-components",
                "COMPONENTS Unknown OPTIONAL_COMPONENTS CUDA",
                "Unknown cgpy component",
                False,
            ),
        ]
        if not cuda:
            cases.append(
                (
                    "required-cuda",
                    "COMPONENTS CUDA",
                    "built without CUDA",
                    False,
                )
            )
        cases.append(("optional-cuda", "OPTIONAL_COMPONENTS CUDA", "", True))
        for name, components, diagnostic, success in cases:
            probe = work / name
            probe.mkdir()
            content = (
                "cmake_minimum_required(VERSION 3.24)\n"
                "project(component_probe LANGUAGES CXX)\n"
                f"find_package(cgpy CONFIG REQUIRED {components})\n"
            )
            if success:
                content += (
                    "if(NOT TARGET cg::core OR TARGET cg::cuda OR cgpy_CUDA_FOUND)\n"
                    '  message(FATAL_ERROR "optional CUDA component contract failed")\n'
                    "endif()\n"
                )
            (probe / "CMakeLists.txt").write_text(content)
            result = subprocess.run(
                [
                    cmake,
                    "-S",
                    str(probe),
                    "-B",
                    str(probe / "build"),
                    *settings,
                    "-DCMAKE_DISABLE_FIND_PACKAGE_CUDAToolkit=ON",
                ],
                cwd=work,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if (
                result.returncode == 0
            ) != success or diagnostic not in result.stdout:
                raise AssertionError(
                    f"component check {name} failed:\n{result.stdout}"
                )
            print(f"Component contract passed: {name}", flush=True)
    print(
        "Relocated C++ consumers passed outside the source tree.", flush=True
    )


if __name__ == "__main__":
    main()
