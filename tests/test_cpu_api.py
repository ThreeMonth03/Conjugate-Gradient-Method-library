"""Public CPU API contracts; numerical kernels are covered by GoogleTest."""

import gc
import warnings

import cgpy
import numpy as np
import pytest


def system(n=9):
    rng = np.random.default_rng(17)
    q = rng.normal(size=(n, n))
    A = q @ q.T + np.eye(n)
    b = rng.normal(size=n)
    return A, b


@pytest.mark.parametrize("algorithm", ["explicit", "standard"])
@pytest.mark.parametrize("fused", [False, True])
def test_solve_preserves_inputs_and_meets_true_residual(algorithm, fused):
    A, b = system()
    x0 = np.ones_like(b)
    original = (A.copy(), b.copy(), x0.copy())
    result = cgpy.solve(A, b, x0, algorithm=algorithm, fused=fused, rtol=1e-10)
    assert result.converged
    assert result.status == "converged"
    assert result.iterations > 0
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-9)
    assert np.linalg.norm(b - A @ result.x) <= 1e-10 * np.linalg.norm(b)
    assert result.residual_norm == pytest.approx(np.linalg.norm(b - A @ result.x), abs=1e-12)
    for expected, actual in zip(original, (A, b, x0), strict=True):
        np.testing.assert_array_equal(actual, expected)


def test_prepared_snapshot_and_workspace_reuse():
    A, b = system()
    expected = np.linalg.solve(A, b)
    solver = cgpy.prepare(A)
    work = solver.create_workspace()
    capacity = work.capacity
    A[:] = np.nan
    del A
    gc.collect()
    for _ in range(3):
        x = np.zeros_like(b)
        info = solver.solve_into(b, x, workspace=work, rtol=1e-10)
        assert info.converged
        assert work.capacity == capacity
        np.testing.assert_allclose(x, expected, rtol=1e-8, atol=1e-9)


def test_convenience_converts_layout_but_into_rejects_it():
    A, b = system()
    assert cgpy.solve(np.asfortranarray(A), b.astype(np.float32)).converged
    with pytest.raises(ValueError):
        cgpy.solve_into(np.asfortranarray(A), b, np.zeros_like(b), workspace=cgpy.Workspace(9))
    solver = cgpy.prepare(A)
    with pytest.raises(ValueError):
        solver.solve_into(b, np.zeros(18)[::2], workspace=solver.create_workspace())
    x = np.zeros_like(b)
    x.setflags(write=False)
    with pytest.raises(ValueError):
        solver.solve_into(b, x, workspace=solver.create_workspace())


def unaligned_copy(array):
    buffer = bytearray(array.nbytes + 1)
    value = np.ndarray(array.shape, dtype=np.float64, buffer=buffer, offset=1)
    value[...] = array
    assert value.flags.c_contiguous and not value.flags.aligned
    return value


@pytest.mark.parametrize("target", ["A", "b", "x"])
def test_strict_calls_reject_unaligned_buffers_and_keep_workspace_usable(target):
    A, b = system()
    x = np.zeros_like(b)
    values = {"A": A, "b": b, "x": x}
    values[target] = unaligned_copy(values[target])
    work = cgpy.Workspace(b.size)
    with pytest.raises(ValueError, match="aligned"):
        cgpy.solve_into(values["A"], values["b"], values["x"], workspace=work)
    assert cgpy.solve_into(A, b, x, workspace=work).converged
    solver = cgpy.prepare(A)
    if target == "A":
        with pytest.raises(ValueError, match="aligned"):
            cgpy.PreparedSolver(values["A"])
    else:
        with pytest.raises(ValueError, match="aligned"):
            solver.solve_into(values["b"], values["x"], workspace=work)


def test_convenience_calls_copy_unaligned_inputs_to_safe_storage():
    A, b = system()
    result = cgpy.solve(unaligned_copy(A), unaligned_copy(b), unaligned_copy(np.zeros_like(b)))
    assert result.converged
    assert result.x.flags.aligned
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-6, atol=1e-8)
    solver = cgpy.prepare(unaligned_copy(A))
    x = np.zeros_like(b)
    assert solver.solve_into(b, x, workspace=solver.create_workspace()).converged


@pytest.mark.parametrize("target", ["A", "b", "x0"])
def test_convenience_rejects_complex_inputs_without_discarding_values(target):
    A, b = system()
    values = {"A": A, "b": b, "x0": np.zeros_like(b)}
    values[target] = values[target].astype(np.complex128) + 1j
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="real"):
            cgpy.solve(**values)
        if target == "A":
            with pytest.raises(ValueError, match="real"):
                cgpy.prepare(values["A"])


@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
@pytest.mark.parametrize("value", [-1, 1.5, 1 << 100])
def test_high_level_iteration_counts_raise_value_error(option, value):
    A, b = system()
    with pytest.raises(ValueError, match=option):
        cgpy.solve(A, b, **{option: value})
    with pytest.raises(ValueError, match=option):
        cgpy.solve_into(A, b, np.zeros_like(b), workspace=cgpy.Workspace(b.size), **{option: value})


def test_native_prepared_iteration_conversion_is_documented_as_type_error():
    A, b = system()
    solver = cgpy.prepare(A)
    with pytest.raises(TypeError):
        solver.solve_into(b, np.zeros_like(b), workspace=solver.create_workspace(), maxiter=-1)


def test_alias_workspace_and_option_validation():
    A, b = system()
    work = cgpy.Workspace(9)
    with pytest.raises(ValueError):
        cgpy.solve_into(A, b, b, workspace=work)
    with pytest.raises(ValueError):
        cgpy.solve_into(A, b, A[0], workspace=work)
    with pytest.raises(ValueError):
        cgpy.solve_into(A, b, np.zeros_like(b), workspace=cgpy.Workspace(8))
    for kwargs in (
        {"rtol": -1.0},
        {"rtol": np.nan},
        {"algorithm": "unknown"},
        {"num_threads": 0},
    ):
        with pytest.raises(ValueError):
            cgpy.solve(A, b, **kwargs)
    # A failed call must release the workspace guard for later solves.
    x = np.zeros_like(b)
    assert cgpy.solve_into(A, b, x, workspace=work).converged


def test_zero_rhs_initial_guess_limits_and_breakdown():
    A = np.eye(5)
    assert cgpy.solve(A, np.zeros(5)).iterations == 0
    b = np.arange(1.0, 6.0)
    assert cgpy.solve(A, b, b).iterations == 0
    limited = cgpy.solve(A, b, maxiter=0)
    assert limited.status == "max_iterations"
    assert limited.iterations == 0
    assert cgpy.solve(-A, b).status == "non_positive_curvature"
    assert cgpy.solve(A, np.full(5, np.nan)).status == "non_finite"


@pytest.mark.skipif(not cgpy.openmp_enabled(), reason="OpenMP backend is disabled")
def test_threads_use_independent_workspaces():
    from concurrent.futures import ThreadPoolExecutor

    A, b = system(65)
    solver = cgpy.prepare(A, num_threads=2)

    def run():
        x = np.zeros_like(b)
        info = solver.solve_into(b, x, workspace=solver.create_workspace(), rtol=1e-10)
        return info, x

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    for info, x in results:
        assert info.converged
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-9)
