"""Jacobi setup, reuse and numerical contracts at the Python boundary."""

import cgpy
import numpy as np
import pytest


@pytest.mark.parametrize("sparse", [False, True])
def test_jacobi_converges_faster_on_diagonally_scaled_spd(sparse):
    n = 16
    diagonal = np.geomspace(1.0, 1e5, n)
    A = (
        cgpy.CSRMatrix(np.arange(n + 1), np.arange(n), diagonal, (n, n))
        if sparse
        else np.diag(diagonal)
    )
    b = diagonal.copy()
    common = {"rtol": 1e-11, "maxiter": 500, "residual_refresh": 0}
    cg = cgpy.solve(A, b, **common)
    pcg = cgpy.solve(A, b, preconditioner="jacobi", **common)
    assert cg.converged and pcg.converged
    assert pcg.iterations < cg.iterations
    assert pcg.iterations <= 2
    np.testing.assert_allclose(pcg.x, np.ones(n), rtol=1e-10, atol=1e-12)
    plan = cgpy.prepare(A, preconditioner="jacobi")
    work = plan.create_workspace()
    assert plan.preconditioner == "jacobi"
    assert plan.preconditioner_setup_ns >= 0
    setup = plan.preconditioner_setup_ns
    for _ in range(3):
        x = np.zeros(n)
        assert plan.solve_into(b, x, workspace=work, **common).converged
        assert plan.preconditioner_setup_ns == setup


@pytest.mark.parametrize(
    "diagonal", [[0.0, 1.0], [-1.0, 1.0], [np.nan, 1.0], [np.inf, 1.0], [1e-320, 1.0]]
)
def test_jacobi_rejects_invalid_diagonal(diagonal):
    with pytest.raises(ValueError):
        cgpy.prepare(np.diag(diagonal), preconditioner="jacobi")
    A = cgpy.CSRMatrix([0, 1, 2], [0, 1], diagonal, (2, 2))
    with pytest.raises(ValueError):
        cgpy.prepare(A, preconditioner="jacobi")


def test_jacobi_requires_diagonal_and_standard_algorithm():
    A = cgpy.CSRMatrix([0, 1, 2], [1, 0], [1.0, 1.0], (2, 2))
    with pytest.raises(ValueError):
        cgpy.prepare(A, preconditioner="jacobi")
    for options in (
        {"preconditioner": "unknown"},
        {"preconditioner": "jacobi", "algorithm": "explicit"},
    ):
        with pytest.raises(ValueError):
            cgpy.solve(np.eye(2), np.ones(2), **options)
    plan = cgpy.prepare(np.eye(2), preconditioner="jacobi")
    x = np.zeros(2)
    work = plan.create_workspace()
    with pytest.raises(ValueError):
        plan.solve_into(np.ones(2), x, workspace=work, algorithm="explicit")
    np.testing.assert_array_equal(x, np.zeros(2))
    assert plan.solve_into(np.ones(2), x, workspace=work).converged


@pytest.mark.parametrize("magnitude", [1e-200, 1e-160, 1e200])
@pytest.mark.parametrize("preconditioner", ["none", "jacobi"])
def test_out_of_range_products_never_report_false_convergence(magnitude, preconditioner):
    import math

    b = np.array([magnitude, -magnitude])
    result = cgpy.solve(np.eye(2), b, preconditioner=preconditioner, rtol=1e-8, atol=0)
    assert not result.converged
    assert result.status == "non_finite"
    stable_norm = math.hypot(*b)
    assert result.residual_norm == pytest.approx(stable_norm, rel=1e-14, abs=0)
    assert result.residual_norm > 1e-8 * stable_norm
