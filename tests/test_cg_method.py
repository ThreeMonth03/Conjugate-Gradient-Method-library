import builtins
import importlib

import cgpy
import numpy as np
import pytest
import utils
from _cgpy import CG
from _cgpy.Matrix import Accelerated_Matrix, Naive_Matrix


def test_linear_cg():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(23, 23))
    a = q @ q.T + np.eye(23)
    b = rng.normal(size=23)
    x0 = rng.normal(size=23)
    expected = np.linalg.solve(a, b)
    implementations = [
        utils.np_linear_CG(x0.copy(), a, b, 1e-10, epoch=1000),
        utils.custom_linear_CG(x=x0.copy(), a=a, b=b, epsilon=1e-10, epoch=1000, num_threads=1),
        utils.custom_linear_CG(x=x0.copy(), a=a, b=b, epsilon=1e-10, epoch=1000, num_threads=2),
    ]
    for actual in implementations:
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
        assert np.linalg.norm(b - a @ actual) <= 1e-9


def test_nonlinear_cg():
    pytest.importorskip("autograd")
    np.random.seed(3)
    x_rand = np.random.uniform(low=3, high=5, size=(2,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        x = np.copy(x_rand)
        print("case 1 np: ", method)
        x, _ = utils.np_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

        print("case 1 custom : ", method, " , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
            1,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

        print("case 1 custom : ", method, " , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

    x_rand = np.random.uniform(low=0.5, high=0.7, size=(100,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        print("case 2 np: ", method)
        x = np.copy(x_rand)
        x, _ = utils.np_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()

        print("case 2 custom: ", method, " , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
            1,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()

        print("case 2 custom: ", method, " , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
            2,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()


if __name__ == "__main__":
    test_linear_cg()
    test_nonlinear_cg()


@pytest.mark.parametrize(
    "matrix_type, method, threads",
    [
        (Naive_Matrix, "solve_by_Naive_Matrix", 1),
        (Accelerated_Matrix, "solve_by_Accelerated_Matrix", 2),
    ],
)
def test_legacy_linear_adapter_matches_explicit_core(matrix_type, method, threads):
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    x0 = np.array([0.5, -0.25])
    inputs = [matrix_type(value) for value in (A, b, x0)]
    solver = CG.linear_CG(1e-10, 30, threads)
    actual = np.array(getattr(solver, method)(*inputs).tolist())
    expected = cgpy.solve(
        A,
        b,
        x0,
        algorithm="explicit",
        rtol=0.0,
        atol=1e-10,
        maxiter=30,
        num_threads=threads if cgpy.openmp_enabled() else 1,
    )
    assert expected.converged
    np.testing.assert_allclose(actual, expected.x, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(inputs[2].tolist(), x0)
    np.testing.assert_array_equal(inputs[0].tolist2d(), A)
    np.testing.assert_array_equal(inputs[1].tolist(), b)
    solver.set_epoch(0)
    np.testing.assert_array_equal(getattr(solver, method)(*inputs).tolist(), x0)


@pytest.mark.parametrize(
    "matrix_type, method",
    [
        (Naive_Matrix, "solve_by_Naive_Matrix"),
        (Accelerated_Matrix, "solve_by_Accelerated_Matrix"),
    ],
)
@pytest.mark.parametrize(
    "epsilon, epochs, threads",
    [(-1.0, 10, 1), (np.nan, 10, 1), (np.inf, 10, 1), (1e-8, -1, 1), (1e-8, 10, 0), (1e-8, 10, -1)],
)
def test_legacy_linear_adapter_rejects_invalid_options(
    matrix_type, method, epsilon, epochs, threads
):
    solver = CG.linear_CG(epsilon, epochs, threads)
    with pytest.raises(ValueError):
        getattr(solver, method)(matrix_type([[2.0]]), matrix_type([4.0]), matrix_type([0.0]))


def test_legacy_autograd_import_is_lazy(monkeypatch):
    original_import = builtins.__import__

    def without_autograd(name, *args, **kwargs):
        if name.split(".")[0] == "autograd":
            raise ImportError("autograd is unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_autograd)
    importlib.reload(utils)
    np.testing.assert_allclose(
        utils.custom_linear_CG(np.zeros(1), np.array([[2.0]]), np.array([4.0]), num_threads=1),
        [2.0],
    )
    with pytest.raises(ImportError, match="autograd is unavailable"):
        utils.grad(lambda x: x * x)


@pytest.mark.parametrize("threads", [0, -1])
def test_legacy_python_wrapper_rejects_invalid_threads(threads):
    with pytest.raises(ValueError, match="num_threads"):
        utils.custom_linear_CG(np.zeros(1), np.array([[2.0]]), np.array([4.0]), num_threads=threads)
