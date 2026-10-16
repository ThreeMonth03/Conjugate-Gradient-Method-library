"""Keep CUDA metadata parsing and fresh-output initialization native."""

from types import SimpleNamespace

import numpy as np
import pytest
from cgpy import cuda
from cgpy._runtime import cuda as cuda_runtime
from cgpy.solver.cuda import api


def test_initial_output_delegates_validation_and_storage_to_native(
    monkeypatch,
):
    class Input:
        @property
        def __cuda_array_interface__(self):
            raise AssertionError("Python must not parse CUDA metadata")

    rhs, initial = Input(), Input()
    native_output = SimpleNamespace(size=3)
    calls = []

    def initial_output(b, x0, device, stream):
        calls.append((b, x0, device, stream))
        return native_output

    monkeypatch.setattr(
        cuda_runtime,
        "load",
        lambda: SimpleNamespace(_initial_output=initial_output),
    )
    output, size = api._initial_output(rhs, initial, 2, 17)
    assert isinstance(output, cuda.Vector)
    assert output._native is native_output and size == 3
    assert calls == [(rhs, initial, 2, 17)]


@pytest.mark.parametrize("has_initial", [False, True])
def test_initial_output_unwraps_owned_vectors(monkeypatch, has_initial):
    rhs = cuda.Vector.__new__(cuda.Vector)
    rhs._native = object()
    initial = cuda.Vector.__new__(cuda.Vector) if has_initial else None
    if initial is not None:
        initial._native = object()
    calls = []

    def initial_output(b, x0, device, stream):
        calls.append((b, x0, device, stream))
        return SimpleNamespace(size=2)

    monkeypatch.setattr(
        cuda_runtime,
        "load",
        lambda: SimpleNamespace(_initial_output=initial_output),
    )
    api._initial_output(rhs, initial, 0, 0)
    assert calls == [
        (rhs._native, initial._native if has_initial else None, 0, 0)
    ]


def test_initial_output_preserves_native_validation_errors(monkeypatch):
    error = ValueError("invalid metadata")

    def initial_output(*args):
        raise error

    monkeypatch.setattr(
        cuda_runtime,
        "load",
        lambda: SimpleNamespace(_initial_output=initial_output),
    )
    with pytest.raises(ValueError) as caught:
        api._initial_output(object(), None, 0, 0)
    assert caught.value is error


@pytest.fixture
def gpu():
    if not cuda.is_available():
        pytest.skip(
            "requires the optional CUDA extension and a compatible GPU"
        )
    return cuda


class Export:
    def __init__(self, descriptor):
        self.__cuda_array_interface__ = descriptor


@pytest.mark.parametrize("entry", ["solve", "session", "strict", "prepared"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("version", None),
        ("version", "3"),
        ("shape", None),
        ("shape", (2, 1)),
        ("shape", (-1,)),
        ("shape", (1 << 100,)),
        ("shape", (2.0,)),
        ("typestr", None),
        ("typestr", "<f4"),
        ("data", None),
        ("data", (0,)),
        ("data", (-1, False)),
        ("data", (0, "readonly")),
        ("strides", ("8",)),
        ("stream", "0"),
    ],
)
def test_cuda_entry_points_share_metadata_error_contract(
    gpu, entry, field, value
):
    matrix = gpu.DenseMatrix.from_host(np.eye(2))
    rhs = gpu.Vector.from_host(np.ones(2))
    descriptor = {**rhs.__cuda_array_interface__, field: value}
    invalid = Export(descriptor)
    plan = gpu.prepare(matrix)
    output = plan.create_vector()
    workspace = plan.create_workspace()
    with pytest.raises(ValueError):
        if entry == "solve":
            gpu.solve(matrix, invalid)
        elif entry == "session":
            gpu.Session(2).solve(matrix, invalid)
        elif entry == "strict":
            gpu.solve_into(matrix, invalid, output, workspace=workspace)
        else:
            plan.solve_into(invalid, output, workspace=workspace)
    np.testing.assert_array_equal(output.to_host(), np.zeros(2))
    assert plan.solve_into(rhs, output, workspace=workspace).converged


@pytest.mark.parametrize("field", ["version", "shape", "typestr", "data"])
def test_cuda_missing_required_metadata_is_value_error(gpu, field):
    matrix = gpu.DenseMatrix.from_host(np.eye(2))
    rhs = gpu.Vector.from_host(np.ones(2))
    descriptor = dict(rhs.__cuda_array_interface__)
    del descriptor[field]
    with pytest.raises(ValueError, match=field):
        gpu.solve(matrix, Export(descriptor))


@pytest.mark.parametrize("descriptor", [None, (), "invalid"])
def test_cuda_interface_requires_a_dictionary(gpu, descriptor):
    with pytest.raises(ValueError, match="dictionary"):
        gpu.solve(object(), Export(descriptor))


@pytest.mark.parametrize("entry", ["solve", "session"])
def test_cuda_initial_guess_is_validated_before_initialization(gpu, entry):
    matrix = gpu.DenseMatrix.from_host(np.eye(2))
    rhs = gpu.Vector.from_host(np.ones(2))
    initial = gpu.Vector.from_host(np.full(3, 4.0))
    solve = gpu.solve if entry == "solve" else gpu.Session(2).solve
    with pytest.raises(ValueError, match="initial guess.*size"):
        solve(matrix, rhs, initial)
    np.testing.assert_array_equal(rhs.to_host(), np.ones(2))
    np.testing.assert_array_equal(initial.to_host(), np.full(3, 4.0))
    assert solve(matrix, rhs).converged


@pytest.mark.parametrize("source", ["getter", "index"])
def test_cuda_preserves_user_metadata_callback_exceptions(gpu, source):
    error = RuntimeError("user callback sentinel")

    class Getter:
        @property
        def __cuda_array_interface__(self):
            raise error

    class Dimension:
        def __index__(self):
            raise error

    rhs = gpu.Vector.from_host(np.ones(2))
    invalid = (
        Getter()
        if source == "getter"
        else Export({**rhs.__cuda_array_interface__, "shape": (Dimension(),)})
    )
    with pytest.raises(RuntimeError) as caught:
        gpu.solve(object(), invalid)
    assert caught.value is error
