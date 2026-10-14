"""Stable verification norms without changing the ordinary CG fast path."""

import math
import sys


def stable_l2(vector):
    """Compute an L2 norm with scaled hypot, including tiny/large components."""
    return math.hypot(*(float(value) for value in vector))


def l2_from_square(squared, vector):
    """Use the usual square root except when squared arithmetic loses range."""
    if squared >= sys.float_info.min and math.isfinite(squared):
        return math.sqrt(squared)
    return stable_l2(vector)
