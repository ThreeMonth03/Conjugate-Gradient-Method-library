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


def normalize_host_array(
    value: npt.ArrayLike,
    ndim: int,
    *,
    name: str = "source",
    copy: bool = False,
) -> npt.NDArray[np.float64]:
    array = normalize_real_array(value, name)
    result = (
        np.array(array, dtype=np.float64, order="C", copy=True)
        if copy
        else np.require(array, dtype=np.float64, requirements=("C", "A"))
    )
    if result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    return result
