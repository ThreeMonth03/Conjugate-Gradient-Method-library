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


@pytest.mark.parametrize(
    "search",
    [utils.np_line_search, utils.custom_naive_line_search, utils.custom_accelerated_line_search],
)
def test_legacy_line_search_bounds_invalid_and_nonfinite_trials(search):
    point = np.array([1.0])
    direction = -point

    def objective(x):
        return float(np.dot(x, x) / 2)

    def derivative(x):
        return np.asarray(x)

    with pytest.raises(ValueError, match="between zero and one"):
        search(objective, derivative, point, direction, beta=1.0)
    calls = []

    def unavailable_away_from_current(x):
        calls.append(x)
        return 0.0 if np.array_equal(x, point) else np.nan

    with pytest.raises(RuntimeError, match="finite decreasing step"):
        search(unavailable_away_from_current, derivative, point, direction)
    assert len(calls) <= utils._MAX_BACKTRACKS + 1
    assert search(objective, derivative, np.zeros(1), np.zeros(1)) == 1.0


def test_accelerated_nonlinear_uses_accelerated_line_search(monkeypatch):
    calls = []
    original = utils.custom_accelerated_line_search

    def accelerated(*args, **kwargs):
        calls.append(kwargs["num_threads"])
        return original(*args, **kwargs)

    def unexpected_naive(*args, **kwargs):
        raise AssertionError("accelerated solver routed to the naive line search")

    monkeypatch.setattr(utils, "custom_accelerated_line_search", accelerated)
    monkeypatch.setattr(utils, "custom_naive_line_search", unexpected_naive)
    point, value = utils.custom_accelerated_nonlinear_CG(
        np.array([1.0, 2.0]),
        1e-10,
        0.1,
        0.8,
        lambda x: np.dot(x, x) / 2,
        lambda x: np.asarray(x),
        num_threads=2,
    )
    np.testing.assert_array_equal(point, np.zeros(2))
    assert value == 0.0 and calls == [2]


@pytest.mark.parametrize(
    "solver",
    [utils.np_nonlinear_CG, utils.custom_naive_nonlinear_CG, utils.custom_accelerated_nonlinear_CG],
)
def test_legacy_nonlinear_initial_convergence_method_validation_and_iteration_limit(
    solver, monkeypatch
):
    def objective(x):
        return np.dot(x, x) / 2

    def derivative(x):
        return np.asarray(x)

    point, value = solver(np.zeros(2), 1e-10, 0.1, 0.8, objective, derivative)
    np.testing.assert_array_equal(point, np.zeros(2))
    assert value == 0.0
    with pytest.raises(AssertionError, match="method not supported"):
        solver(np.ones(2), 1e-10, 0.1, 0.8, objective, derivative, method="unsupported")
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        solver(np.ones(2), 1e-10, 0.1, 0.8, objective, lambda x: np.full(2, np.nan))
    monkeypatch.setattr(utils, "_MAX_NONLINEAR_ITERATIONS", 2)
    with pytest.raises(RuntimeError, match="iteration limit"):
        solver(np.zeros(1), 1e-10, 0.1, 0.8, lambda x: -x[0], lambda x: np.array([-1.0]))


def test_legacy_nonlinear_rejects_nan_line_search_step(monkeypatch):
    monkeypatch.setattr(utils, "np_line_search", lambda *args, **kwargs: np.nan)
    with pytest.raises(RuntimeError, match="invalid line-search step"):
        utils.np_nonlinear_CG(
            np.ones(1),
            1e-10,
            0.1,
            0.8,
            lambda x: x[0] ** 2,
            lambda x: 2 * x,
        )


def test_legacy_numpy_linear_does_not_false_converge_on_tiny_rhs():
    with pytest.raises(RuntimeError, match="finite positive curvature"):
        utils.np_linear_CG(np.zeros(2), np.eye(2), np.array([1e-200, -1e-200]), epsilon=1e-208)


@pytest.mark.parametrize(
    "solver",
    [utils.np_nonlinear_CG, utils.custom_naive_nonlinear_CG, utils.custom_accelerated_nonlinear_CG],
)
@pytest.mark.parametrize("initial", [0.0, 1.0])
def test_legacy_nonlinear_does_not_false_converge_on_tiny_gradient(solver, initial):
    def objective(x):
        return x[0] ** 2 / 2 + 1e-200 * x[0]

    def derivative(x):
        return np.array([x[0] + 1e-200])

    with pytest.raises(RuntimeError, match="descent direction"):
        solver(np.array([initial]), 1e-208, 0.1, 0.8, objective, derivative)
