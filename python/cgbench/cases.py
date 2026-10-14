"""Deterministic inputs shared by numerical tests and performance runs."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CaseSpec:
    """Describe a dense, symmetric, strictly diagonally dominant SPD system."""

    name: str
    size: int
    condition_scale: float = 100.0

    def to_dict(self):
        return asdict(self)


SUITES = {
    "smoke": (CaseSpec("dense-16", 16),),
    "cpu": tuple(CaseSpec(f"dense-{size}", size) for size in (64, 256, 1024, 2048)),
}


def make_case(spec, seed=20261015):
    """Build A, b, x0 and a known solution outside the measured call.

    A symmetric matrix plus a diagonal exceeding its absolute off-diagonal row
    sum is positive definite. ``condition_scale`` controls the diagonal range;
    it is deliberately not advertised as the matrix's exact condition number.
    """
    import numpy as np

    if spec.size < 1 or spec.condition_scale < 1:
        raise ValueError("size and condition_scale must be positive")
    generator = np.random.default_rng(seed)
    raw = generator.standard_normal((spec.size, spec.size))
    matrix = (raw + raw.T) / (2.0 * spec.size)
    np.fill_diagonal(matrix, 0.0)
    diagonal = np.abs(matrix).sum(axis=1)
    diagonal += np.geomspace(1.0, spec.condition_scale, spec.size)
    np.fill_diagonal(matrix, diagonal)
    solution = generator.standard_normal(spec.size)
    rhs = matrix @ solution
    initial = np.zeros(spec.size, dtype=np.float64)
    return matrix, rhs, initial, solution


def input_hash(spec, arrays):
    """Hash the specification and exact logical bytes, including dtype/shape."""
    digest = hashlib.sha256(json.dumps(spec.to_dict(), sort_keys=True).encode())
    for array in arrays:
        digest.update(array.dtype.str.encode())
        digest.update(json.dumps(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
