"""Validate solver shape and Python iteration-count contracts."""

import operator

import numpy as np

_MAX_SIZE_T = np.iinfo(np.uintp).max


def validate_iteration_limits(
    maxiter: int, residual_refresh: int
) -> tuple[int, int]:
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


def validate_operator_shape(shape: tuple[int, ...]) -> None:
    if len(shape) != 2 or shape[0] == 0 or shape[0] != shape[1]:
        raise ValueError("A must be a nonempty square matrix")
