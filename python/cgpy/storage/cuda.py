"""Owned CUDA vector storage and explicit synchronous transfers."""

from __future__ import annotations

import typing

import numpy as np
import numpy.typing as npt

from .._interop import host
from .._runtime import cuda as cuda_runtime
from .._typing import SupportsCudaArrayInterface

if typing.TYPE_CHECKING:
    from .. import _cuda


class Vector:
    """Own a writable float64 GPU vector, initialized to zero.

    Parameters
    ----------
    size : int
        Nonnegative number of elements.
    device : int, default=0
        Device ordinal, fixed for the allocation's lifetime.
    stream : int, default=0
        Borrowed stream for initialization. All methods finish their work
        before returning; CUDA Array Interface exports have no pending work.

    Notes
    -----
    ``__cuda_array_interface__`` permits zero-copy interoperability. Its
    consumer must retain this owner and avoid conflicting mutation. Inputs
    from PyTorch must be detached float64 contiguous CUDA tensors. Allocation
    and host transfers belong outside a prepared-solve timing interval.
    """

    _native: _cuda.Vector

    def __init__(self, size: int, *, device: int = 0, stream: int = 0) -> None:
        self._native = cuda_runtime.load().Vector(size, device, stream)
        self.fill(0.0, stream=stream)

    @classmethod
    def _empty(cls, size: int, device: int, stream: int) -> typing.Self:
        result = cls.__new__(cls)
        result._native = cuda_runtime.load().Vector(size, device, stream)
        return result

    @classmethod
    def from_host(
        cls, source: npt.ArrayLike, *, device: int = 0, stream: int = 0
    ) -> typing.Self:
        """Snapshot a real host vector, normalizing and uploading it once."""
        source = host.normalize_host_array(source, 1)
        result = cls._empty(source.size, device, stream)
        result._native.copy_from_host(source, stream)
        return result

    @property
    def size(self) -> int:
        """int: Number of allocated float64 elements."""
        return self._native.size

    @property
    def shape(self) -> tuple[int]:
        """tuple: One-dimensional shape."""
        return (self.size,)

    @property
    def device(self) -> int:
        """int: Device ordinal of the owned allocation."""
        return self._native.device

    @property
    def __cuda_array_interface__(self) -> dict[str, object]:
        """dict: Version-3 device view, with all producer work complete."""
        return self._native.__cuda_array_interface__

    def fill(self, value: float, *, stream: int = 0) -> None:
        """Fill the vector on a borrowed stream and wait for completion."""
        self._native.fill(value, stream)

    def copy_from_host(
        self, source: npt.ArrayLike, *, stream: int = 0
    ) -> None:
        """Normalize and synchronously upload a matching host vector."""
        self._native.copy_from_host(
            host.normalize_host_array(source, 1), stream
        )

    def copy_from_device(
        self, source: SupportsCudaArrayInterface, *, stream: int = 0
    ) -> None:
        """Copy a matching contiguous float64 device vector on a stream.

        CUDA Array Interface version 3 producer dependencies are honored. A
        version-2 producer, including PyTorch, must be ready on this same
        stream or explicitly synchronized by the caller before the copy.
        """
        self._native.copy_from_device(
            source._native if isinstance(source, Vector) else source, stream
        )

    def to_host(self, *, stream: int = 0) -> npt.NDArray[np.float64]:
        """Synchronously download an independent NumPy float64 snapshot."""
        return self._native.to_host(stream)
