"""Compatibility helpers for linear and nonlinear conjugate gradient methods."""

from functools import partial

import numpy as np
from _cgpy import CG
from _cgpy.Matrix import Accelerated_Matrix, Naive_Matrix

_MAX_BACKTRACKS = 256
_MAX_NONLINEAR_ITERATIONS = 10000
_METHOD_SUFFIXES = {
    "Fletcher_Reeves": "Fletcher_Reeves",
    "Hager-Zhang": "Hager_Zhang",
    "Dai-Yuan": "Dai_Yuan",
}


def grad(*args, **kwargs):
    """Construct an autograd derivative when the optional dependency is used."""
    from autograd import grad as autograd_grad

    return autograd_grad(*args, **kwargs)


def is_pos_def(x):
    """Return whether all eigenvalues are positive."""
    return np.all(np.linalg.eigvals(x) > 0)


def generate_pos_def(n):
    """Generate a random Gram matrix of dimension n."""
    A = np.random.rand(n, n)
    return A.dot(A.T)


def generate_symmetric(A):
    """Return the symmetric part of A."""
    return (A + A.T) / 2


def custom_linear_CG(x, a, b, epsilon=5e-7, epoch=10000000, num_threads=3):
    """Solve Ax=b using the legacy absolute tolerance and return a NumPy vector."""
    if num_threads <= 0:
        raise ValueError("num_threads must be positive")
    matrix_type = Naive_Matrix if num_threads == 1 else Accelerated_Matrix
    solver = CG.linear_CG(epsilon, epoch, num_threads)
    solve = solver.solve_by_Naive_Matrix if num_threads == 1 else solver.solve_by_Accelerated_Matrix
    return np.array(solve(matrix_type(a), matrix_type(b), matrix_type(x)).tolist())


def np_linear_CG(x, A, b, epsilon, epoch=10000000):
    """Solve Ax=b with the legacy explicit-residual NumPy formulation."""
    if not np.isfinite(epsilon) or epsilon < 0 or epoch < 0:
        raise ValueError("epsilon and epoch must be finite and nonnegative")
    residual = A.dot(x) - b
    direction = -residual
    for _ in range(epoch):
        if not np.all(np.isfinite(residual)):
            raise RuntimeError("Linear CG encountered a non-finite residual")
        if np.linalg.norm(residual) <= epsilon:
            return x
        product = A.dot(direction)
        denominator = direction.dot(product)
        if not np.isfinite(denominator) or denominator <= 0:
            raise RuntimeError("Linear CG requires finite positive curvature")
        step = -residual.dot(direction) / denominator
        x = x + step * direction
        residual = A.dot(x) - b
        coefficient = residual.dot(product) / denominator
        direction = coefficient * direction - residual
    if not np.all(np.isfinite(residual)):
        raise RuntimeError("Linear CG encountered a non-finite residual")
    return x


def nonlinear_func_1(x):
    """Return the two-variable example whose minimum is at (1, 1)."""
    return x[0] ** 4 - 2 * x[0] ** 2 * x[1] + x[0] ** 2 + x[1] ** 2 - 2 * x[0] + 1


def nonlinear_func_2(x):
    """Return sqrt(sum(x**4)), the higher-dimensional legacy example."""
    x = np.array(x)
    return np.sum(x**4) ** 0.5


def _line_search(f, df, x, d, alpha, beta, trial):
    if not (0 < alpha < 1 and 0 < beta < 1):
        raise ValueError("alpha and beta must lie strictly between zero and one")
    slope = float(np.dot(df(x), d))
    value = float(f(x))
    if not np.isfinite(slope) or not np.isfinite(value):
        raise RuntimeError("Line search requires a finite objective and gradient")
    if slope == 0 and np.all(np.asarray(d) == 0):
        return 1.0
    if slope >= 0:
        raise RuntimeError("Line search requires a descent direction")
    step = 1.0
    for _ in range(_MAX_BACKTRACKS):
        candidate = trial(step)
        candidate_value = float(f(candidate))
        if np.isfinite(candidate_value) and candidate_value <= value + alpha * step * slope:
            return step
        if np.array_equal(candidate, x):
            break
        step *= beta
    raise RuntimeError("Line search failed to find a finite decreasing step")


def custom_naive_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """Find an Armijo step with Naive_Matrix vector operations."""
    point, direction = Naive_Matrix(x), Naive_Matrix(d)
    return _line_search(f, df, x, d, alpha, beta, lambda step: (point + direction * step).tolist())


def custom_accelerated_line_search(f, df, x, d, alpha=5e-4, beta=0.8, num_threads=1):
    """Find an Armijo step with Accelerated_Matrix vector operations."""
    if num_threads == -1:
        num_threads = 1
    if num_threads < 1:
        raise ValueError("num_threads must be positive or -1")
    point, direction = Accelerated_Matrix(x), Accelerated_Matrix(d)
    point.set_number_of_threads(num_threads)
    direction.set_number_of_threads(num_threads)
    return _line_search(f, df, x, d, alpha, beta, lambda step: (point + direction * step).tolist())


def np_line_search(f, df, x, d, alpha=5e-4, beta=0.8):
    """Find an Armijo step with NumPy vector operations."""
    return _line_search(f, df, x, d, alpha, beta, lambda step: x + step * d)


def _supported_method(method):
    if method not in _METHOD_SUFFIXES:
        raise AssertionError("method not supported")
    return _METHOD_SUFFIXES[method]


def _native_direction(method, matrix_type, num_threads=1):
    suffix = _supported_method(method)
    accelerated = matrix_type is Accelerated_Matrix
    prefix = "Accelerated" if accelerated else "Naive"
    update = getattr(CG.nonlinear_CG, f"{prefix}_{suffix}_next_iteration")

    def next_direction(current, following, direction):
        arguments = [matrix_type(value) for value in (current, following, direction)]
        result = update(*arguments, num_threads) if accelerated else update(*arguments)
        return np.array(result.tolist())

    return next_direction


def _nonlinear_cg(X, tol, alpha, beta, f, Df, update, line_search):
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    point = np.array(X, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(point)):
        raise ValueError("The initial point must contain finite values")

    def objective(x):
        value = float(f(x))
        if not np.isfinite(value):
            raise RuntimeError("Nonlinear CG encountered a non-finite objective")
        return value

    def gradient(x):
        value = np.asarray(Df(x), dtype=np.float64)
        if value.shape != point.shape:
            raise ValueError("Gradient shape must match the current point")
        if not np.all(np.isfinite(value)):
            raise RuntimeError("Nonlinear CG encountered a non-finite gradient")
        return value

    current = gradient(point)
    direction = -current
    for _ in range(_MAX_NONLINEAR_ITERATIONS):
        if np.linalg.norm(current) < tol:
            return point, objective(point)
        if not np.all(np.isfinite(direction)) or np.dot(current, direction) >= 0:
            direction = -current
        step = line_search(f, Df, point, direction, alpha, beta)
        if step is None:
            return point, objective(point)
        if not np.isfinite(step) or step <= 0:
            raise RuntimeError("Nonlinear CG received an invalid line-search step")
        following_point = point + step * direction
        if not np.all(np.isfinite(following_point)):
            raise RuntimeError("Nonlinear CG encountered a non-finite point")
        if np.array_equal(following_point, point):
            raise RuntimeError("Nonlinear CG stagnated before reaching the gradient tolerance")
        following = gradient(following_point)
        if np.linalg.norm(following) < tol:
            return following_point, objective(following_point)
        direction = np.asarray(update(current, following, direction), dtype=np.float64)
        point, current = following_point, following
    raise RuntimeError("Nonlinear CG exceeded its iteration limit")


def custom_nonlinear_CG(X, tol, alpha, beta, f, Df, method="Fletcher_Reeves", num_threads=-1):
    """Return (x, f(x)) using the selected legacy native direction update."""
    if num_threads == 1:
        return custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method)
    return custom_accelerated_nonlinear_CG(X, tol, alpha, beta, f, Df, method, num_threads)


def custom_naive_nonlinear_CG(X, tol, alpha, beta, f, Df, method="Fletcher_Reeves"):
    """Run nonlinear CG with native Naive_Matrix direction updates."""
    return _nonlinear_cg(
        X,
        tol,
        alpha,
        beta,
        f,
        Df,
        _native_direction(method, Naive_Matrix),
        custom_naive_line_search,
    )


def custom_accelerated_nonlinear_CG(
    X, tol, alpha, beta, f, Df, method="Fletcher_Reeves", num_threads=-1
):
    """Run nonlinear CG with native accelerated updates and line search."""
    if num_threads != -1 and num_threads < 1:
        raise ValueError("num_threads must be positive or -1")
    return _nonlinear_cg(
        X,
        tol,
        alpha,
        beta,
        f,
        Df,
        _native_direction(method, Accelerated_Matrix, num_threads),
        partial(custom_accelerated_line_search, num_threads=num_threads),
    )


def np_nonlinear_CG(X, tol, alpha, beta, f, Df, method="Fletcher_Reeves"):
    """Run nonlinear CG with NumPy direction updates and line search."""
    _supported_method(method)
    methods = {
        "Fletcher_Reeves": Fletcher_Reeves_next_iteration,
        "Hager-Zhang": Hager_Zhang_next_iteration,
        "Dai-Yuan": Dai_Yuan_next_iteration,
    }
    return _nonlinear_cg(X, tol, alpha, beta, f, Df, methods[method], np_line_search)


def Fletcher_Reeves_next_iteration(cur_Df, next_Df, delta):
    """Update the direction with the Fletcher-Reeves coefficient."""
    coefficient = np.linalg.norm(next_Df) ** 2 / np.linalg.norm(cur_Df) ** 2
    return -next_Df + coefficient * delta


def Hager_Zhang_next_iteration(cur_Df, next_Df, delta):
    """Update the direction with the Hager-Zhang coefficient."""
    difference = next_Df - cur_Df
    projection = delta.dot(difference)
    corrected = difference - delta * np.linalg.norm(difference) ** 2 * 2 / projection
    coefficient = corrected.dot(next_Df / projection)
    return -next_Df + coefficient * delta


def Dai_Yuan_next_iteration(cur_Df, next_Df, delta):
    """Update the direction with the Dai-Yuan coefficient."""
    coefficient = np.linalg.norm(next_Df) ** 2 / delta.dot(next_Df - cur_Df)
    return -next_Df + coefficient * delta
