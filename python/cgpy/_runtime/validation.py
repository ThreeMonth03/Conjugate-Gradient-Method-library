"""Validate Python values before conversion or native calls."""

import operator

import numpy as np
import numpy.typing as npt

_MAX_SIZE_T = np.iinfo(np.uintp).max


def real_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(
            f"{name} must contain real values; complex inputs are unsupported"
        )
    return array


def dense_matrix(A: npt.ArrayLike) -> np.ndarray:
    A = np.require(
        real_array(A, "A"), dtype=np.float64, requirements=("C", "A")
    )
    if A.ndim != 2 or A.shape[0] == 0 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a nonempty square matrix")
    return A


def iteration_limits(maxiter: int, residual_refresh: int) -> tuple[int, int]:
    counts = []
    for name, value in (
        ("maxiter", maxiter),
        ("residual_refresh", residual_refresh),
    ):
        try:
            count = operator.index(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a nonnegative integer") from exc
        if count < 0 or count > _MAX_SIZE_T:
            raise ValueError(f"{name} must fit a nonnegative size_t")
        counts.append(count)
    return counts[0], counts[1]


def host_array(value: npt.ArrayLike, ndim: int) -> np.ndarray:
    result = np.require(
        real_array(value, "source"), dtype=np.float64, requirements=("C", "A")
    )
    if result.ndim != ndim:
        raise ValueError(f"source must have {ndim} dimensions")
    return result
