"""Borrow native CUDA owners through public Python façades."""

from ..matrix import cuda as matrix
from ..storage import cuda as storage


def native_object(value: object) -> object:
    return (
        value._native
        if isinstance(
            value, (storage.Vector, matrix.DenseMatrix, matrix.CsrMatrix)
        )
        else value
    )
