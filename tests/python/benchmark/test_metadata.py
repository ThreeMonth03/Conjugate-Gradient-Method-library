"""Attribute native measurements to their actual CMake source checkout."""

import hashlib
import sys
from types import SimpleNamespace

import pytest
from cgbench import artifacts


def loaded_core(binary, monkeypatch):
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"loaded native extension")
    monkeypatch.setitem(
        sys.modules, "cgpy._core", SimpleNamespace(__file__=str(binary))
    )
    monkeypatch.setitem(
        sys.modules,
        "cgpy",
        SimpleNamespace(__file__=str(binary.parent / "__init__.py")),
    )


def test_external_build_directory_uses_its_cmake_source_checkout(
    tmp_path, monkeypatch
):
    source = tmp_path / "baseline-worktree"
    source.mkdir()
    build = tmp_path / "external-build"
    binary = build / "python/cgpy/_core.so"
    loaded_core(binary, monkeypatch)
    (build / "CMakeCache.txt").write_text(
        f"CMAKE_HOME_DIRECTORY:INTERNAL={source}\n"
        "CMAKE_BUILD_TYPE:STRING=Release\n"
        "CMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG\n"
    )
    # The harness and extension may come from completely different locations.
    monkeypatch.setattr(
        artifacts, "__file__", str(tmp_path / "harness/artifacts.py")
    )
    calls = []

    def git(arguments, cwd):
        calls.append((arguments, cwd))
        assert cwd == source
        return "baseline-revision" if arguments[1] == "rev-parse" else ""

    monkeypatch.setattr(artifacts, "_command", git)
    metadata = artifacts.environment_metadata()
    assert metadata["source_root"] == str(source)
    assert metadata["revision"] == "baseline-revision"
    assert metadata["git_status"] == ""
    assert len(calls) == 2
    assert metadata["cmake"]["path"] == str(build / "CMakeCache.txt")
    assert (
        metadata["native"]["module_sha256"]
        == hashlib.sha256(binary.read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("cache", ("unrelated", "missing-source", "absent"))
def test_unknown_source_does_not_fall_back_to_another_checkout(
    cache, tmp_path, monkeypatch
):
    if cache == "unrelated":
        binary = tmp_path / ".venv/lib/site-packages/cgpy/_core.so"
        loaded_core(binary, monkeypatch)
        (tmp_path / "CMakeCache.txt").write_text(
            f"CMAKE_HOME_DIRECTORY:INTERNAL={tmp_path}\n"
        )
    else:
        binary = tmp_path / "python/cgpy/_core.so"
        loaded_core(binary, monkeypatch)
        if cache == "missing-source":
            (tmp_path / "CMakeCache.txt").write_text(
                "CMAKE_BUILD_TYPE:STRING=Release\n"
            )

    def forbidden(*args):
        raise AssertionError(
            "unknown source queried an unrelated Git checkout"
        )

    monkeypatch.setattr(artifacts, "_command", forbidden)
    metadata = artifacts.environment_metadata()
    assert metadata["source_root"] is None
    assert metadata["revision"] is None
    assert metadata["git_status"] is None
    assert (metadata["cmake"] is None) == (cache != "missing-source")
