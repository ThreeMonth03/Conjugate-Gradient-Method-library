"""Keep optional CUDA loads retryable without retaining duplicate wrappers."""

from collections import Counter
from types import SimpleNamespace

import pytest
from cgpy._runtime import cuda as runtime


@pytest.fixture
def loader(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime, "_extension", None)
    monkeypatch.setattr(runtime, "_libraries", {})
    monkeypatch.setattr(
        runtime.importlib.util, "find_spec", lambda _: object()
    )
    distribution = SimpleNamespace(locate_file=lambda name: tmp_path / name)
    monkeypatch.setattr(
        runtime.importlib.metadata, "distribution", lambda _: distribution
    )
    paths = [
        tmp_path / "nvidia/cuda_runtime/lib/libcudart.so.12",
        tmp_path / "nvidia/cublas/lib/libcublasLt.so.12",
        tmp_path / "nvidia/cublas/lib/libcublas.so.12",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    calls = []

    def open_library(path, *, mode):
        assert mode == runtime.ctypes.RTLD_GLOBAL
        calls.append(path)
        return object()

    monkeypatch.setattr(runtime.ctypes, "CDLL", open_library)
    return SimpleNamespace(paths=paths, calls=calls, open_library=open_library)


def test_extension_failure_reuses_libraries_and_success_is_memoized(
    loader, monkeypatch
):
    failure = ImportError("extension dependency is unavailable")
    attempts = []
    extension = object()

    def import_extension(name):
        attempts.append(name)
        if len(attempts) <= 3:
            raise failure
        return extension

    monkeypatch.setattr(runtime.importlib, "import_module", import_extension)
    for _ in range(3):
        with pytest.raises(RuntimeError, match="CUDA extension") as caught:
            runtime.load()
        assert caught.value.__cause__ is failure
        assert runtime._extension is None
        assert len(runtime._libraries) == 3
    assert loader.calls == [str(path) for path in loader.paths]
    assert runtime.load() is extension

    def unexpected_lookup(*args):
        raise AssertionError("a loaded extension must bypass discovery")

    monkeypatch.setattr(runtime.importlib.util, "find_spec", unexpected_lookup)
    assert runtime.load() is extension
    assert attempts == ["cgpy._cuda"] * 4
    assert loader.calls == [str(path) for path in loader.paths]


def test_failed_library_retries_after_partial_success(loader, monkeypatch):
    failure = OSError("library dependency is unavailable")
    attempts = Counter()
    extension = object()

    def open_library(path, *, mode):
        attempts[path] += 1
        if path == str(loader.paths[1]) and attempts[path] == 1:
            raise failure
        return loader.open_library(path, mode=mode)

    monkeypatch.setattr(runtime.ctypes, "CDLL", open_library)
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: extension
    )
    with pytest.raises(RuntimeError, match="libcublasLt.so.12") as caught:
        runtime.load()
    assert caught.value.__cause__ is failure
    assert set(runtime._libraries) == {loader.paths[0]}
    assert runtime._extension is None
    assert runtime.load() is extension
    assert attempts == {
        str(loader.paths[0]): 1,
        str(loader.paths[1]): 2,
        str(loader.paths[2]): 1,
    }
    assert set(runtime._libraries) == set(loader.paths)


@pytest.mark.parametrize("missing", ["distribution", "file"])
def test_newly_available_runtime_is_discovered_after_failure(
    loader, monkeypatch, missing
):
    available = False
    distribution = runtime.importlib.metadata.distribution
    extension = object()

    def find_distribution(name):
        if missing == "distribution" and not available:
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return distribution(name)

    def import_extension(name):
        if not available:
            raise OSError("runtime dependency is unavailable")
        return extension

    monkeypatch.setattr(
        runtime.importlib.metadata, "distribution", find_distribution
    )
    monkeypatch.setattr(runtime.importlib, "import_module", import_extension)
    if missing == "file":
        for path in loader.paths:
            path.unlink()
    with pytest.raises(RuntimeError, match="CUDA extension"):
        runtime.load()
    assert not runtime._libraries
    assert not loader.calls
    available = True
    for path in loader.paths:
        path.touch()
    assert runtime.load() is extension
    assert loader.calls == [str(path) for path in loader.paths]


def test_newly_built_extension_is_discovered_after_missing_spec(
    loader, monkeypatch
):
    extension = object()
    monkeypatch.setattr(runtime.importlib.util, "find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="CG_ENABLE_CUDA=ON"):
        runtime.load()
    assert not loader.calls
    monkeypatch.setattr(
        runtime.importlib.util, "find_spec", lambda _: object()
    )
    monkeypatch.setattr(
        runtime.importlib, "import_module", lambda _: extension
    )
    assert runtime.load() is extension
    assert loader.calls == [str(path) for path in loader.paths]
