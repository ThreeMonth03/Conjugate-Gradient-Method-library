"""Assemble a pinned Linux x86-64 CUDA 12.8 toolchain inside this checkout."""

import argparse
import concurrent.futures
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
COMPONENTS = (
    (
        "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz",
        "9961b3484b6b71314063709a4f9529654f96782ad39e72bf1e00f070db8210d3",
    ),
    (
        "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz",
        "8d566b5fe745c46842dc16945cf36686227536decd2302c372be86da37faca68",
    ),
    (
        "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz",
        "0740e9e01e4f15e17c5ab8d68bba4f8ec0eb6b84edccba4ac45112d2d2174e4b",
    ),
)
RUNTIMES = {"nvidia-cuda-runtime-cu12": "12.8.90", "nvidia-cublas-cu12": "12.8.4.1"}
SANITIZER = (
    "cuda_sanitizer_api/linux-x86_64/cuda_sanitizer_api-linux-x86_64-12.8.93-archive.tar.xz",
    "ae3574f052c0e06c95305962668eb1fe6ab571dfbb58b305fdb14d523bb1b240",
)


def digest(path):
    """Hash a downloaded component without reading it all into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(component, cache):
    """Fetch one fixed archive and verify its published SHA-256 digest."""
    relative, expected = component
    target = cache / Path(relative).name
    if target.exists() and digest(target) == expected:
        return target
    partial = target.with_suffix(target.suffix + ".partial")
    print(f"Downloading {relative}", flush=True)
    with urllib.request.urlopen(BASE + relative, timeout=120) as source, partial.open("wb") as dest:
        shutil.copyfileobj(source, dest)
    if digest(partial) != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"CUDA component checksum mismatch: {relative}")
    partial.replace(target)
    return target


def runtime_sources(prefix, cache):
    """Use public NVIDIA wheels, installing only into a local target if needed."""
    installed = {}
    for name, version in RUNTIMES.items():
        try:
            package = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            break
        if package.version != version:
            break
        directory = "cuda_runtime" if "runtime" in name else "cublas"
        installed[directory] = Path(package.locate_file(f"nvidia/{directory}")).resolve()
    if len(installed) == len(RUNTIMES):
        return installed
    target = prefix / "runtime"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--upgrade",
            "--cache-dir",
            str(cache.parent / "pip"),
            "--target",
            str(target),
            *(f"{name}=={version}" for name, version in RUNTIMES.items()),
        ],
        check=True,
    )
    return {name: target / "nvidia" / name for name in ("cuda_runtime", "cublas")}


def merge_component(source, destination):
    """Merge a component without following preexisting links into another environment."""
    if source.is_symlink():
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        destination.symlink_to(source.readlink())
    elif source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            merge_component(child, destination / child.name)
    else:
        if destination.is_symlink():
            destination.unlink()
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--prefix", type=Path, default=root / ".deps/cuda-12.8")
    parser.add_argument(
        "--sanitizer", action="store_true", help="also install local Compute Sanitizer"
    )
    args = parser.parse_args()
    prefix = args.prefix.resolve()
    if sys.platform != "linux" or platform.machine() not in ("x86_64", "AMD64"):
        parser.error("the pinned bootstrap supports Linux x86-64")
    if not prefix.is_relative_to(root) or prefix == root:
        parser.error("--prefix must be a dependency directory inside the checkout")
    cache = root / ".cache/cuda"
    cache.mkdir(parents=True, exist_ok=True)
    prefix.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        components = COMPONENTS + ((SANITIZER,) if args.sanitizer else ())
        archives = list(pool.map(lambda entry: download(entry, cache), components))
    for archive in archives:
        with tempfile.TemporaryDirectory(prefix="extract-", dir=cache) as temporary:
            with tarfile.open(archive) as contents:
                contents.extractall(temporary, filter="data")
            directories = list(Path(temporary).iterdir())
            if len(directories) != 1 or not directories[0].is_dir():
                raise RuntimeError(f"unexpected CUDA component layout: {archive}")
            component = directories[0]
            for item in component.iterdir():
                if item.is_dir():
                    merge_component(item, prefix / item.name)
                else:
                    # Each upstream license remains attributable to its component.
                    destination = prefix / "licenses" / archive.name.removesuffix(".tar.xz")
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination / item.name)
    runtimes = runtime_sources(prefix, cache)
    (prefix / "lib").mkdir(exist_ok=True)
    (prefix / "include").mkdir(exist_ok=True)
    for source in runtimes.values():
        for header in (source / "include").glob("*.h"):
            shutil.copy2(header, prefix / "include" / header.name)
        for library in (source / "lib").glob("*.so*"):
            destination = prefix / "lib" / library.name
            if destination.is_symlink() or destination.exists():
                destination.unlink()
            destination.symlink_to(os.path.relpath(library, destination.parent))
    for name in ("cublas", "cublasLt"):
        link = prefix / "lib" / f"lib{name}.so"
        if not link.exists():
            link.symlink_to(f"lib{name}.so.12")
    library_alias = prefix / "lib64"
    if not library_alias.exists():
        library_alias.symlink_to("lib", target_is_directory=True)
    (prefix / "version.json").write_text(json.dumps({"cuda": {"version": "12.8.1"}}) + "\n")
    subprocess.run([str(prefix / "bin/nvcc"), "--version"], check=True)
    print(f"Local CUDA toolchain: {prefix}")
    print("Pass CUDAToolkit_ROOT and CMAKE_CUDA_COMPILER explicitly; no shell files were changed.")


if __name__ == "__main__":
    main()
