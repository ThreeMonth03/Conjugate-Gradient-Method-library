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
    A = cgpy.CsrMatrix(indptr, indices, values, dense.shape)
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
        cgpy.CsrMatrix(indptr, indices, values, (2, 2))


def test_csr_input_alias_and_nonfinite_status():
    indptr, indices, values, dense = sparse_system()
    A = cgpy.CsrMatrix(indptr, indices, values, dense.shape)
    b = np.ones(3)
    with pytest.raises(ValueError):
        cgpy.solve_into(A, b, b, workspace=cgpy.Workspace(3))
    values[0] = np.nan
    bad = cgpy.CsrMatrix(indptr, indices, values, dense.shape)
    assert cgpy.solve(bad, b).status == "non_finite"


@pytest.mark.parametrize(
    "shape, indptr, indices, data",
    [
        ((2, 3), [0, 1, 2], [0, 1], [1.0, 1.0]),
        ((0, 0), [0], [], []),
    ],
)
def test_prepare_requires_nonempty_square_csr(shape, indptr, indices, data):
    A = cgpy.CsrMatrix(indptr, indices, data, shape)
    with pytest.raises(ValueError, match="nonempty square"):
        cgpy.prepare(A)
    with pytest.raises(ValueError, match="nonempty square"):
        cgpy.solve(A, np.zeros(shape[0]))


@pytest.mark.parametrize(
    "dtype",
    [
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    ],
)
def test_csr_index_buffer_width_and_sign_preserve_the_operator(dtype):
    indptr, indices, values, dense = sparse_system()
    A = cgpy.CsrMatrix(
        indptr.astype(dtype), indices.astype(dtype), values, dense.shape
    )
    rhs = dense @ np.array([1.0, 2.0, 3.0])
    answer = cgpy.solve(A, rhs, rtol=1e-12)
    assert answer.converged
    np.testing.assert_allclose(answer.x, [1.0, 2.0, 3.0])


def foreign_buffer(array, layout):
    if layout == "strided":
        backing = np.empty(array.size * 2, dtype=array.dtype)
        backing[::2] = array
        return backing[::2]
    if layout == "reversed":
        return array[::-1].copy()[::-1]
    if layout == "unaligned":
        result = np.ndarray(
            array.shape,
            dtype=array.dtype,
            buffer=bytearray(array.nbytes + 1),
            offset=1,
        )
        result[:] = array
        return result
    if layout == "byte_swapped":
        return array.astype(array.dtype.newbyteorder("S"))
    result = array.copy()
    result.setflags(write=False)
    return result


@pytest.mark.parametrize(
    "layout", ["strided", "reversed", "unaligned", "byte_swapped", "readonly"]
)
def test_csr_foreign_buffers_are_normalized_and_snapshotted(layout):
    indptr, indices, values, dense = sparse_system()
    inputs = [
        foreign_buffer(array, layout) for array in (indptr, indices, values)
    ]
    A = cgpy.CsrMatrix(*inputs, dense.shape)
    for array in inputs:
        array.setflags(write=True)
        array.fill(0)
    answer = cgpy.solve(A, dense @ np.ones(3), rtol=1e-12)
    assert answer.converged
    np.testing.assert_allclose(answer.x, np.ones(3))


@pytest.mark.parametrize("dtype", [np.int16, np.float32, np.float64])
def test_csr_real_values_convert_to_float64_without_changing_the_operator(
    dtype,
):
    indptr, indices, values, dense = sparse_system()
    A = cgpy.CsrMatrix(indptr, indices, values.astype(dtype), dense.shape)
    answer = cgpy.solve(A, dense @ np.ones(3), rtol=1e-12)
    assert answer.converged and answer.x.dtype == np.float64
    np.testing.assert_allclose(answer.x, np.ones(3))


def test_csr_unsigned_indices_are_not_wrapped_through_signed_integer_types():
    maximum = int(np.iinfo(np.uintp).max)
    large = maximum - 1
    A = cgpy.CsrMatrix(
        [0, 1], np.array([large], dtype=np.uint64), [1.0], (1, maximum)
    )
    assert A.shape == (1, maximum) and A.nnz == 1
    with pytest.raises(ValueError, match="range"):
        cgpy.CsrMatrix(
            [0, 1], np.array([maximum], dtype=np.uint64), [1.0], (1, maximum)
        )
    with pytest.raises(ValueError, match="endpoints"):
        cgpy.CsrMatrix(
            np.array([0, maximum], dtype=np.uint64), [0], [1.0], (1, 1)
        )


@pytest.mark.parametrize("dtype", [np.int8, np.int64])
def test_csr_negative_indices_are_rejected_before_unsigned_conversion(dtype):
    with pytest.raises(ValueError, match="nonnegative"):
        cgpy.CsrMatrix([0, 1], np.array([-1], dtype=dtype), [1.0], (1, 1))


def test_csr_empty_indices_and_complex_value_rejection():
    A = cgpy.CsrMatrix([0], [], [], (0, 0))
    assert A.shape == (0, 0) and A.nnz == 0
    with pytest.raises(ValueError, match="real"):
        cgpy.CsrMatrix([0, 1], [0], [1 + 1j], (1, 1))
    with pytest.raises(TypeError):
        cgpy.CsrMatrix([0], [], [], (0, 1 << 100))
