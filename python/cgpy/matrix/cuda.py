"""Immutable device matrix snapshots, independent of solver policy."""

from __future__ import annotations

import typing

import numpy.typing as npt

from .._interop import host
from .._runtime import cuda as cuda_runtime
from . import csr

if typing.TYPE_CHECKING:
    from .. import _cuda


class DenseMatrix:
    """Own an immutable dense float64 device snapshot.

    Construct with ``from_host`` or ``from_device``. The matrix has no public
    mutable storage view; prepared solvers take a further owning snapshot.
    """

    _native: _cuda.DenseMatrix

    def __init__(self) -> None:
        raise TypeError("use DenseMatrix.from_host or DenseMatrix.from_device")

    @classmethod
    def from_host(
        cls, source: npt.ArrayLike, *, device: int = 0, stream: int = 0
    ) -> typing.Self:
        """Normalize a real 2D host array and explicitly upload a snapshot."""
        result = cls.__new__(cls)
        result._native = cuda_runtime.load().DenseMatrix.from_host(
            host.normalize_host_array(source, 2), device, stream
        )
        return result

    @classmethod
    def from_device(
        cls, source: object, *, device: int = 0, stream: int = 0
    ) -> typing.Self:
        """Snapshot a C-contiguous float64 CUDA Array Interface matrix.

        Version-2 producers must be ready on the specified stream. Version-3
        producer stream dependencies are honored. The call is synchronous.
        """
        result = cls.__new__(cls)
        result._native = cuda_runtime.load().DenseMatrix.from_device(
            source, device, stream
        )
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """tuple: Matrix dimensions (rows, columns)."""
        return self._native.shape

    @property
    def device(self) -> int:
        """int: Device ordinal of the snapshot."""
        return self._native.device


class CsrMatrix:
    """Own an immutable canonical CSR device snapshot.

    Columns must be sorted, unique and in range in each row; pointers must be
    monotone, start at zero and end at nnz. Setup validates the indices once;
    it does not silently sort or coalesce them. Use native CUDA CSR kernels.
    """

    _native: _cuda.CsrMatrix

    def __init__(self) -> None:
        raise TypeError("use CsrMatrix.from_host or CsrMatrix.from_device")

    @classmethod
    def from_host(
        cls, source: csr.CsrMatrix, *, device: int = 0, stream: int = 0
    ) -> typing.Self:
        """Explicitly upload a public CPU cgpy.CsrMatrix snapshot."""
        if not isinstance(source, csr.CsrMatrix):
            raise TypeError("source must be cgpy.CsrMatrix")
        result = cls.__new__(cls)
        result._native = cuda_runtime.load().CsrMatrix.from_host(
            source._native, device, stream
        )
        return result

    @classmethod
    def from_device(
        cls,
        indptr: object,
        indices: object,
        data: object,
        shape: tuple[int, int],
        *,
        device: int = 0,
        stream: int = 0,
    ) -> typing.Self:
        """Validate canonical structure and snapshot device CSR arrays.

        Pointers and columns must use matching signed int32 or int64 types.
        Values must be float64. All three arrays must be contiguous on the
        selected device. A setup-time index download is included in this call.
        """
        result = cls.__new__(cls)
        result._native = cuda_runtime.load().CsrMatrix.from_device(
            indptr, indices, data, shape, device, stream
        )
        return result

    @property
    def shape(self) -> tuple[int, int]:
        """tuple: Matrix dimensions (rows, columns)."""
        return self._native.shape

    @property
    def nnz(self) -> int:
        """int: Number of stored entries."""
        return self._native.nnz

    @property
    def device(self) -> int:
        """int: Device ordinal of the snapshot."""
        return self._native.device
