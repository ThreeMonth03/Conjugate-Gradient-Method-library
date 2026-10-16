"""Canonical owning compressed sparse row matrices."""

import numpy as np
import numpy.typing as npt

from .. import _core
from .._interop import host


def _normalize_index_array(
    value: npt.ArrayLike, *, allow_empty: bool = False
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError("CSR index arrays must be one-dimensional")
    if array.dtype.kind not in "iu":
        if allow_empty and array.size == 0:
            array = np.empty(0, dtype=np.uintp)
        else:
            raise ValueError("CSR index arrays must contain integers")
    # Preserve width/sign so the native copy can reject negative or overflowing
    # indices before converting to size_t. Normalize only foreign byte order,
    # alignment and strides; native contiguous buffers remain borrowed here.
    return np.require(
        array,
        dtype=array.dtype.newbyteorder("="),
        requirements=("C", "A"),
    )


class CsrMatrix:
    """Own a validated canonical compressed sparse row matrix.

    Parameters
    ----------
    indptr : array_like of int
        One-dimensional row pointers of length ``shape[0] + 1``. They must
        start at zero, be nondecreasing, and end at the number of values.
    indices : array_like of int
        One-dimensional nonnegative column indices, sorted and unique within
        each row. Each entry must be smaller than ``shape[1]``.
    data : array_like of float
        One-dimensional real values matching ``indices``. Values are copied
        into float64 storage.
    shape : tuple of int
        Nonnegative row and column counts. Solving additionally requires a
        nonempty square matrix.

    Raises
    ------
    ValueError
        Index arrays, dimensions, canonical ordering or data shape are invalid.
    TypeError
        Native dimensions or indices cannot be represented by ``size_t``.

    Notes
    -----
    Construction snapshots the arrays. Later changes to them do not alter the
    operator. Strided, unaligned, read-only and non-native-endian inputs are
    normalized when needed. Signed/unsigned integer indices are range-checked
    before native conversion. Canonicalize SciPy CSR with ``sum_duplicates()`` and
    ``sort_indices()`` before wrapping arrays that need those operations.

    Examples
    --------
    >>> import cgpy
    >>> A = cgpy.CsrMatrix([0, 2, 4], [0, 1, 0, 1], [4., 1., 1., 3.], (2, 2))
    >>> A.shape, A.nnz
    ((2, 2), 4)
    """

    def __init__(
        self,
        indptr: npt.ArrayLike,
        indices: npt.ArrayLike,
        data: npt.ArrayLike,
        shape: tuple[int, int],
    ) -> None:
        rowptr = _normalize_index_array(indptr)
        columns = _normalize_index_array(indices, allow_empty=True)
        values = host.normalize_host_array(data, 1, name="data")
        self._native = _core.CsrMatrix(rowptr, columns, values, shape)

    @property
    def shape(self) -> tuple[int, int]:
        """tuple of int: Matrix dimensions ``(rows, columns)``."""
        return self._native.shape

    @property
    def nnz(self) -> int:
        """int: Number of stored entries, including explicitly stored zeros."""
        return self._native.nnz
