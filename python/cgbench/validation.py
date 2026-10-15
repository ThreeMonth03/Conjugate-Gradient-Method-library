"""Validate numerical evidence independently of artifact or timing schemas."""

import math
import numbers
import sys


def validate_converged_result(result):
    """Require finite true-residual convergence and valid operation counts."""
    if result.get("solver_status") != "converged":
        raise ValueError("measured result must be converged")
    residual = result.get("true_residual_norm")
    threshold = result.get("residual_threshold")
    for value in (residual, threshold):
        try:
            valid = (
                isinstance(value, numbers.Real)
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            )
        except OverflowError:
            valid = False
        if not valid:
            raise ValueError(
                "true residual and threshold must be finite and nonnegative"
            )
    if residual > threshold * (1 + 64 * sys.float_info.epsilon):
        raise ValueError("measured result failed the true-residual gate")
    if any(
        type(result.get(name)) is not int or result[name] < 0
        for name in ("iterations", "matvecs")
    ):
        raise ValueError("operation counts must be nonnegative integers")
