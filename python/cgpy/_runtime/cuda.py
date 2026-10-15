"""Load the optional CUDA extension and inspect explicit capabilities."""

import ctypes
import importlib
import importlib.metadata
import importlib.util
import pathlib

_extension = None
_libraries = []


def load():
    global _extension
    if _extension is not None:
        return _extension
    if importlib.util.find_spec("cgpy._cuda") is None:
        raise RuntimeError("CUDA requires a cgpy build with CG_ENABLE_CUDA=ON")
    # These are public NVIDIA distributions, independent of PyTorch's layout.
    for package, directory, names in (
        ("nvidia-cuda-runtime-cu12", "cuda_runtime", ("libcudart.so.12",)),
        (
            "nvidia-cublas-cu12",
            "cublas",
            ("libcublasLt.so.12", "libcublas.so.12"),
        ),
    ):
        try:
            distribution = importlib.metadata.distribution(package)
        except importlib.metadata.PackageNotFoundError:
            # A developer's explicitly configured loader can supply CUDA.
            continue
        for name in names:
            path = pathlib.Path(
                distribution.locate_file(f"nvidia/{directory}/lib/{name}")
            )
            if path.is_file():
                try:
                    _libraries.append(
                        ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                    )
                except OSError as error:
                    raise RuntimeError(
                        f"CUDA runtime library could not load: {name}"
                    ) from error
    try:
        _extension = importlib.import_module("cgpy._cuda")
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "CUDA extension could not load; install the cgpy[cuda] runtimes"
        ) from error
    return _extension


def enabled() -> bool:
    """Return whether the CUDA extension loads, without requiring a GPU."""
    try:
        load()
    except (RuntimeError, OSError):
        return False
    return True


def is_available(device: int = 0) -> bool:
    """Return whether the extension loads and CUDA exposes the requested GPU.

    A disabled extension or absent/incompatible driver returns False. This
    capability query never selects a different backend or device. The GPU must
    also support an architecture included in the explicitly configured build.
    """
    return enabled() and bool(load().is_available(device))


def build_info() -> dict[str, object]:
    """Return CUDA build metadata or the reason the extension is disabled."""
    try:
        return dict(load().build_info())
    except (RuntimeError, OSError) as error:
        return {"enabled": False, "reason": str(error)}


def runtime_info(device: int = 0) -> dict[str, object]:
    """Describe the GPU, compute capability and CUDA runtime versions.

    ``driver_api_version`` is the driver's supported CUDA API version, rather
    than NVIDIA's display-driver release number. Raises RuntimeError when CUDA
    or the requested device is unavailable.
    """
    return dict(load().runtime_info(device))


def synchronize(device: int = 0, stream: int = 0) -> None:
    """Wait for a CUDA stream and restore the caller's current device.

    Parameters
    ----------
    device : int, default=0
        Explicit device ordinal.
    stream : int, default=0
        Borrowed CUDA stream handle on that device; zero is the legacy default.
        The caller must keep a nondefault stream alive until the call returns.
    """
    load().synchronize(device, stream)
