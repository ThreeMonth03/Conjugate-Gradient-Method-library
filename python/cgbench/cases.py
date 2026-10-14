"""Deterministic inputs shared by numerical tests and performance runs."""

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CaseSpec:
    """Describe a dense SPD system or a CSR Dirichlet Poisson grid."""

    name: str
    size: int
    condition_scale: float = 100.0
    storage: str = "dense"
    grid: int | None = None
    dimensions: int = 2
    diagonal_scale: float = 1.0

    def to_dict(self):
        data = asdict(self)
        # Preserve existing dense artifact hashes when adding sparse cases.
        if self.storage == "dense":
            data.pop("storage")
        if self.grid is None:
            data.pop("grid")
        if self.dimensions == 2:
            data.pop("dimensions")
        if self.diagonal_scale == 1.0:
            data.pop("diagonal_scale")
        return data


SUITES = {
    "smoke": (CaseSpec("dense-16", 16),),
    "cpu": tuple(CaseSpec(f"dense-{size}", size) for size in (64, 256, 1024, 2048)),
    "smoke-csr": (CaseSpec("poisson-4", 16, storage="csr", grid=4),),
    "cpu-csr-f64-v1": tuple(
        CaseSpec(f"poisson-{grid}", grid * grid, storage="csr", grid=grid)
        for grid in (8, 32, 64, 128)
    ),
}
SUITES["cpu-dense-f64-v1"] = SUITES["cpu"]
SUITES["sparse-solvers"] = tuple(
    CaseSpec(
        f"scaled-poisson-{dimensions}d-{grid}",
        grid**dimensions,
        storage="csr",
        grid=grid,
        dimensions=dimensions,
        diagonal_scale=20.0,
    )
    for dimensions, grid in ((2, 16), (2, 32), (3, 6), (3, 10))
)
SUITES["smoke-sparse-solvers"] = tuple(
    CaseSpec(
        f"scaled-poisson-{dimensions}d-{grid}",
        grid**dimensions,
        storage="csr",
        grid=grid,
        dimensions=dimensions,
        diagonal_scale=10.0,
    )
    for dimensions, grid in ((2, 4), (3, 3))
)
SUITES["sparse-solvers"] += (CaseSpec("poisson-2d-16-neutral", 256, storage="csr", grid=16),)
SUITES["smoke-sparse-solvers"] += (CaseSpec("poisson-2d-4-neutral", 16, storage="csr", grid=4),)
SUITES["sparse-solvers-large"] = tuple(
    CaseSpec(
        f"poisson-{dimensions}d-{grid}-large",
        grid**dimensions,
        storage="csr",
        grid=grid,
        dimensions=dimensions,
    )
    for dimensions, grid in ((2, 64), (2, 128), (3, 20), (3, 30))
)


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
    if spec.storage == "csr":
        from scipy import sparse

        if (
            spec.grid is None
            or spec.grid < 1
            or spec.dimensions not in (2, 3)
            or spec.grid**spec.dimensions != spec.size
        ):
            raise ValueError("CSR Poisson size must equal grid to dimensions (2 or 3)")
        line = sparse.diags(
            (-np.ones(spec.grid - 1), 2.0 * np.ones(spec.grid), -np.ones(spec.grid - 1)),
            (-1, 0, 1),
            shape=(spec.grid, spec.grid),
            format="csr",
        )
        matrix = line
        for _ in range(1, spec.dimensions):
            matrix = sparse.kronsum(matrix, line, format="csr")
        if spec.diagonal_scale < 1.0 or not np.isfinite(spec.diagonal_scale):
            raise ValueError("diagonal_scale must be finite and at least one")
        if spec.diagonal_scale != 1.0:
            weights = np.geomspace(1.0, spec.diagonal_scale, spec.size)
            matrix.data *= np.repeat(weights, np.diff(matrix.indptr)) * weights[matrix.indices]
        matrix.sum_duplicates()
        matrix.eliminate_zeros()
        matrix.sort_indices()
    elif spec.storage == "dense":
        raw = generator.standard_normal((spec.size, spec.size))
        matrix = (raw + raw.T) / (2.0 * spec.size)
        np.fill_diagonal(matrix, 0.0)
        diagonal = np.abs(matrix).sum(axis=1)
        diagonal += np.geomspace(1.0, spec.condition_scale, spec.size)
        np.fill_diagonal(matrix, diagonal)
    else:
        raise ValueError("storage must be dense or csr")
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
        if hasattr(array, "indptr"):
            digest.update(b"csr")
            for component in (array.indptr, array.indices, array.data):
                digest.update(component.dtype.str.encode())
                digest.update(component.tobytes(order="C"))
        else:
            digest.update(array.tobytes(order="C"))
    return digest.hexdigest()
