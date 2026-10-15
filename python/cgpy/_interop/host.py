"""Normalize real host arrays without imposing solver-specific shapes."""

import numpy as np
import numpy.typing as npt


def normalize_real_array(value: npt.ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(
            f"{name} must contain real values; complex inputs are unsupported"
        )
    return array


def normalize_dense_matrix(A: npt.ArrayLike) -> np.ndarray:
    A = np.require(
        normalize_real_array(A, "A"), dtype=np.float64, requirements=("C", "A")
    )
    if A.ndim != 2:
        raise ValueError("A must be a two-dimensional matrix")
    return A


def normalize_host_array(value: npt.ArrayLike, ndim: int) -> np.ndarray:
    result = np.require(
        normalize_real_array(value, "source"),
        dtype=np.float64,
        requirements=("C", "A"),
    )
    if result.ndim != ndim:
        raise ValueError(f"source must have {ndim} dimensions")
    return result
