"""Canonical CSR storage and the shared public CPU solver contract."""

import cgpy
import numpy as np
import pytest


def sparse_system():
    indptr = np.array([0, 2, 5, 7], dtype=np.int32)
    indices = np.array([0, 1, 0, 1, 2, 1, 2], dtype=np.int32)
    values = np.array([4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 3.0])
    dense = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    return indptr, indices, values, dense


@pytest.mark.parametrize("algorithm", ["explicit", "standard"])
@pytest.mark.parametrize("fused", [False, True])
def test_csr_matches_dense_and_owns_snapshot(algorithm, fused):
    indptr, indices, values, dense = sparse_system()
    A = cgpy.CSRMatrix(indptr, indices, values, dense.shape)
    values[:] = 0
    indices[:] = 0
    indptr[:] = 0
    b = np.array([1.0, 2.0, 3.0])
    assert A.shape == (3, 3)
    assert A.nnz == 7
    result = cgpy.solve(A, b, algorithm=algorithm, fused=fused, rtol=1e-12)
    assert result.converged
    np.testing.assert_allclose(
        result.x, np.linalg.solve(dense, b), rtol=1e-10, atol=1e-12
    )
    plan = cgpy.prepare(A)
    work = plan.create_workspace()
    for _ in range(2):
        x = np.zeros_like(b)
        info = plan.solve_into(
            b, x, workspace=work, algorithm=algorithm, fused=fused, rtol=1e-12
        )
        assert info.converged
        assert np.linalg.norm(b - dense @ x) <= 1e-12 * np.linalg.norm(b)


@pytest.mark.parametrize(
    "indptr, indices, values",
    [
        ([0, 2, 1], [0], [1.0]),
        ([0, 1, 2], [0, 2], [1.0, 1.0]),
        ([0, 2, 2], [0, 0], [1.0, 1.0]),
        ([0, 2, 2], [1, 0], [1.0, 1.0]),
        ([0, 1, 1], [-1], [1.0]),
        ([0.0, 1.0, 1.0], [0], [1.0]),
        ([0, 1, 1], [0], [[1.0]]),
    ],
)
def test_csr_rejects_malformed_storage(indptr, indices, values):
    with pytest.raises(ValueError):
        cgpy.CSRMatrix(indptr, indices, values, (2, 2))


def test_csr_input_alias_and_nonfinite_status():
    indptr, indices, values, dense = sparse_system()
    A = cgpy.CSRMatrix(indptr, indices, values, dense.shape)
    b = np.ones(3)
    with pytest.raises(ValueError):
        cgpy.solve_into(A, b, b, workspace=cgpy.Workspace(3))
    values[0] = np.nan
    bad = cgpy.CSRMatrix(indptr, indices, values, dense.shape)
    assert cgpy.solve(bad, b).status == "non_finite"


@pytest.mark.parametrize(
    "shape, indptr, indices, data",
    [
        ((2, 3), [0, 1, 2], [0, 1], [1.0, 1.0]),
        ((0, 0), [0], [], []),
    ],
)
def test_prepare_requires_nonempty_square_csr(shape, indptr, indices, data):
    A = cgpy.CSRMatrix(indptr, indices, data, shape)
    with pytest.raises(ValueError, match="nonempty square"):
        cgpy.prepare(A)
    with pytest.raises(ValueError, match="nonempty square"):
        cgpy.solve(A, np.zeros(shape[0]))
