"""Borrow native CUDA owners through public Python façades."""

from typing import overload

from .._typing import SupportsCudaArrayInterface
from ..matrix import cuda as matrix
from ..storage import cuda as storage


@overload
def native_object(
    value: SupportsCudaArrayInterface,
) -> SupportsCudaArrayInterface: ...


@overload
def native_object(value: object) -> object: ...


def native_object(value: object) -> object:
    return (
        value._native
        if isinstance(
            value, (storage.Vector, matrix.DenseMatrix, matrix.CsrMatrix)
        )
        else value
    )
