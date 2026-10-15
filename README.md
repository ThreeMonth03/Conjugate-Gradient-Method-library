# Conjugate Gradient Method Library

A C++20 library with Python bindings for solving real symmetric
positive-definite systems using CG and Jacobi PCG.

- Dense and canonical CSR matrices with float64 arithmetic.
- Prepared solves and reusable workspaces for repeated systems.
- Native CPU kernels, optional OpenMP and dense BLAS.
- Optional CUDA backend with native kernels and dense cuBLAS.

[Documentation](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/)
· [Architecture](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/architecture.html)
· [Benchmarks](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/benchmarks.html)

## Quick start

Building from source requires **CPython 3.12.x** and a C++20 compiler.
Run from the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip .
```

```python
import numpy as np
import cgpy

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
result = cgpy.solve(A, b, preconditioner="jacobi", rtol=1e-10)
assert result.converged
np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])
```

Convergence is verified against the true residual. Symmetry and positive
definiteness are caller preconditions. See the
[API guide](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/api.html)
for sparse matrices, prepared solves and ownership rules.

## Development

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip \
    -r requirements/test.lock -r requirements/docs.lock
make test lint docs doctest
```

See the [build guide](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/build.html)
for optional backends, sanitizer and distribution checks, and
[coding conventions](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/style.html) for coding conventions.
