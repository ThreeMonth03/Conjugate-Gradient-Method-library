"""Public CPU API contracts; numerical kernels are covered by GoogleTest."""

import gc
import warnings

import cgpy
import numpy as np
import pytest
from cgpy import _core


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
    np.testing.assert_allclose(
        result.x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-9
    )
    assert np.linalg.norm(b - A @ result.x) <= 1e-10 * np.linalg.norm(b)
    assert result.residual_norm == pytest.approx(
        np.linalg.norm(b - A @ result.x), abs=1e-12
    )
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
        cgpy.solve_into(
            np.asfortranarray(A),
            b,
            np.zeros_like(b),
            workspace=cgpy.Workspace(9),
        )
    solver = cgpy.prepare(A)
    with pytest.raises(ValueError):
        solver.solve_into(
            b, np.zeros(18)[::2], workspace=solver.create_workspace()
        )
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
def test_strict_calls_reject_unaligned_buffers_and_keep_workspace_usable(
    target,
):
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
            _core.prepare(values["A"])
    else:
        with pytest.raises(ValueError, match="aligned"):
            solver.solve_into(values["b"], values["x"], workspace=work)


@pytest.mark.parametrize("entry", ["strict", "prepared"])
@pytest.mark.parametrize("target", ["b", "x"])
@pytest.mark.parametrize(
    "invalid", ["float32", "strided", "matrix", "byte_swapped"]
)
def test_strict_vector_conversion_rejects_layout_without_mutation(
    entry, target, invalid
):
    A, b = system()
    x = np.zeros_like(b)
    bad = {
        "float32": np.zeros(b.size, dtype=np.float32),
        "strided": np.zeros(2 * b.size)[::2],
        "matrix": np.zeros((1, b.size)),
        "byte_swapped": np.zeros(
            b.size, dtype=np.dtype("float64").newbyteorder("S")
        ),
    }[invalid]
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    values = {"b": b, "x": x}
    values[target] = bad
    before = bad.copy()
    with pytest.raises(ValueError, match="one-dimensional float64"):
        if entry == "strict":
            cgpy.solve_into(A, **values, workspace=workspace)
        else:
            plan.solve_into(**values, workspace=workspace)
    np.testing.assert_array_equal(bad, before)
    np.testing.assert_array_equal(x, np.zeros_like(b))
    assert plan.solve_into(b, x, workspace=workspace).converged


@pytest.mark.parametrize("entry", ["strict", "prepared"])
def test_strict_views_accept_readonly_matrix_and_rhs(entry):
    A, b = system()
    A.setflags(write=False)
    b.setflags(write=False)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)
    info = (
        cgpy.solve_into(A, b, x, workspace=workspace)
        if entry == "strict"
        else plan.solve_into(b, x, workspace=workspace)
    )
    assert info.converged
    assert not A.flags.writeable and not b.flags.writeable
    np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-6, atol=1e-8)


def test_convenience_calls_copy_unaligned_inputs_to_safe_storage():
    A, b = system()
    result = cgpy.solve(
        unaligned_copy(A), unaligned_copy(b), unaligned_copy(np.zeros_like(b))
    )
    assert result.converged
    assert result.x.flags.aligned
    np.testing.assert_allclose(
        result.x, np.linalg.solve(A, b), rtol=1e-6, atol=1e-8
    )
    solver = cgpy.prepare(unaligned_copy(A))
    x = np.zeros_like(b)
    assert solver.solve_into(
        b, x, workspace=solver.create_workspace()
    ).converged


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
@pytest.mark.parametrize("value", [-1, 1.5, 1 << 100, None, "1"])
def test_public_iteration_counts_raise_value_error_and_preserve_workspace(
    option, value
):
    A, b = system()
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)
    with pytest.raises(ValueError, match=option):
        cgpy.solve(A, b, **{option: value})
    with pytest.raises(ValueError, match=option):
        cgpy.solve_into(
            A,
            b,
            x,
            workspace=workspace,
            **{option: value},
        )
    with pytest.raises(ValueError, match=option):
        plan.solve_into(b, x, workspace=workspace, **{option: value})
    np.testing.assert_array_equal(x, np.zeros_like(b))
    assert plan.solve_into(b, x, workspace=workspace).converged


class IndexCount:
    def __init__(self, value):
        self.value = value
        self.calls = 0

    def __index__(self):
        self.calls += 1
        return self.value


@pytest.mark.parametrize("entry", ["convenience", "strict", "prepared"])
@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
def test_cpu_integer_protocol_is_evaluated_once_per_option(entry, option):
    A, b = np.eye(2), np.ones(2)
    count = IndexCount(2)
    options = {option: count}
    if entry == "convenience":
        assert cgpy.solve(A, b, **options).converged
    else:
        plan = cgpy.prepare(A)
        workspace = plan.create_workspace()
        x = np.zeros_like(b)
        info = (
            cgpy.solve_into(A, b, x, workspace=workspace, **options)
            if entry == "strict"
            else plan.solve_into(b, x, workspace=workspace, **options)
        )
        assert info.converged
    assert count.calls == 1


@pytest.mark.parametrize("entry", ["strict", "prepared"])
@pytest.mark.parametrize("option", ["maxiter", "residual_refresh"])
def test_cpu_validates_output_after_integer_protocol_callbacks(entry, option):
    A, b = np.eye(2), np.ones(2)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)

    class FreezeOutput(IndexCount):
        def __index__(self):
            x.setflags(write=False)
            return super().__index__()

    count = FreezeOutput(2)
    with pytest.raises(ValueError, match="writable"):
        if entry == "strict":
            cgpy.solve_into(A, b, x, workspace=workspace, **{option: count})
        else:
            plan.solve_into(b, x, workspace=workspace, **{option: count})
    assert count.calls == 1
    np.testing.assert_array_equal(x, np.zeros_like(b))
    x.setflags(write=True)
    assert plan.solve_into(b, x, workspace=workspace).converged


@pytest.mark.parametrize(
    "count", [np.int64(2), np.uint64(2), IndexCount(2), True]
)
def test_all_cpu_entry_points_accept_integer_index_protocol(count):
    A, b = np.eye(2), np.ones(2)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    assert cgpy.solve(A, b, maxiter=count, residual_refresh=count).converged
    x = np.zeros_like(b)
    assert cgpy.solve_into(
        A, b, x, workspace=workspace, maxiter=count, residual_refresh=count
    ).converged
    x.fill(0)
    assert plan.solve_into(
        b, x, workspace=workspace, maxiter=count, residual_refresh=count
    ).converged


@pytest.mark.parametrize("count", [False, IndexCount(0), np.uint64(0)])
def test_zero_update_limit_has_same_meaning_for_all_cpu_entry_points(count):
    A, b = np.eye(2), np.ones(2)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)
    results = [
        cgpy.solve(A, b, maxiter=count).info,
        cgpy.solve_into(A, b, x, workspace=workspace, maxiter=count),
        plan.solve_into(b, x, workspace=workspace, maxiter=count),
    ]
    assert all(info.status == "max_iterations" for info in results)
    assert all(info.iterations == 0 for info in results)
    np.testing.assert_array_equal(x, np.zeros_like(b))


def test_native_size_t_max_is_accepted_without_truncating_iteration_counts():
    maximum = int(np.iinfo(np.uintp).max)
    A, b = np.eye(2), np.zeros(2)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)
    options = {"maxiter": maximum, "residual_refresh": maximum}
    assert cgpy.solve(A, b, **options).converged
    assert cgpy.solve_into(A, b, x, workspace=workspace, **options).converged
    assert plan.solve_into(b, x, workspace=workspace, **options).converged


@pytest.mark.parametrize("error", [MemoryError, RuntimeError])
def test_iteration_index_protocol_preserves_unexpected_exceptions(error):
    class BrokenIndex:
        def __index__(self):
            raise error("index sentinel")

    A, b = np.eye(2), np.ones(2)
    plan = cgpy.prepare(A)
    workspace = plan.create_workspace()
    x = np.zeros_like(b)
    for call in (
        lambda: cgpy.solve(A, b, maxiter=BrokenIndex()),
        lambda: cgpy.solve_into(
            A, b, x, workspace=workspace, maxiter=BrokenIndex()
        ),
        lambda: plan.solve_into(
            b, x, workspace=workspace, maxiter=BrokenIndex()
        ),
    ):
        with pytest.raises(error, match="index sentinel"):
            call()
    np.testing.assert_array_equal(x, np.zeros_like(b))
    assert plan.solve_into(b, x, workspace=workspace).converged


def test_convenience_normalizes_foreign_storage_and_owns_initial_guess_copy():
    values = np.array([1.0, 99.0, 2.0, 99.0], dtype=">f4")
    initial = values[::2]
    initial.setflags(write=False)
    A = np.asfortranarray(np.eye(2, dtype=np.float32))
    b = np.array([2.0, 1.0], dtype=np.float32)[::-1]
    b.setflags(write=False)
    answer = cgpy.solve(A, b, initial)
    assert answer.converged and answer.iterations == 0
    assert answer.x.dtype == np.float64
    assert answer.x.flags.c_contiguous and answer.x.flags.aligned
    assert answer.x.flags.owndata and answer.x.flags.writeable
    assert not np.shares_memory(answer.x, initial)
    answer.x.fill(0)
    np.testing.assert_array_equal(initial, [1.0, 2.0])
    np.testing.assert_array_equal(b, [1.0, 2.0])


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


@pytest.mark.skipif(
    not cgpy.openmp_enabled(), reason="OpenMP backend is disabled"
)
def test_threads_use_independent_workspaces():
    from concurrent.futures import ThreadPoolExecutor

    A, b = system(65)
    solver = cgpy.prepare(A, num_threads=2)

    def run():
        x = np.zeros_like(b)
        info = solver.solve_into(
            b, x, workspace=solver.create_workspace(), rtol=1e-10
        )
        return info, x

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    for info, x in results:
        assert info.converged
        np.testing.assert_allclose(
            x, np.linalg.solve(A, b), rtol=1e-8, atol=1e-9
        )


@pytest.mark.parametrize("storage", ("dense", "csr"))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
def test_prepared_type_and_resource_contract_is_independent_of_matrix_storage(
    storage, preconditioner
):
    dense = np.array([[4.0, 1.0], [1.0, 3.0]])
    operator = (
        dense
        if storage == "dense"
        else cgpy.CsrMatrix([0, 2, 4], [0, 1, 0, 1], dense.ravel(), (2, 2))
    )
    with pytest.raises(TypeError, match="constructor"):
        cgpy.PreparedSolver(operator, preconditioner=preconditioner)
    plan = cgpy.prepare(operator, preconditioner=preconditioner)
    assert type(plan) is cgpy.PreparedSolver
    assert plan.shape == (2, 2)
    assert plan.size == 2
    assert plan.preconditioner == preconditioner
    assert plan.preconditioner_nbytes == (
        16 if preconditioner == "jacobi" else 0
    )
    assert plan.preconditioner_setup_ns >= 0
    workspace = plan.create_workspace()
    assert workspace.size == plan.size
    b = np.array([1.0, 2.0])
    x = np.zeros_like(b)
    info = plan.solve_into(b, x, workspace=workspace)
    assert info.converged
    np.testing.assert_allclose(x, np.linalg.solve(dense, b))
    assert not hasattr(cgpy, "PreparedCsrSolver")
    assert not hasattr(cgpy, "CSRMatrix")
