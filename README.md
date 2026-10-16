# Conjugate Gradient Method Library

A C++20 library with CPython 3.12 bindings for solving real symmetric
positive-definite systems using CG and Jacobi PCG. Dense and canonical CSR
matrices use float64 arithmetic on CPU or an explicit optional CUDA backend.

- A native core independent of Python, with storage, kernels and solver
  coordination separated by feature.
- RAII owners and borrowed views make lifetimes explicit; prepared solves,
  workspaces and CUDA sessions offer distinct forms of reuse.
- True-residual convergence checks and shared numerical policies, with
  separate host/device execution flows.

[Documentation](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/)
· [Architecture](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/architecture.html)
· [API choices](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/#api-choices)
· [Performance and limits](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/benchmarks.html)

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

Symmetry and positive definiteness are caller preconditions. See the
[API guide](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/api.html)
for sparse matrices, strict inputs and ownership rules.

## Build and contribute

The [build guide](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/build.html)
covers optional OpenMP, BLAS and CUDA backends, tests and documentation.
[Native C++ installation](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/install.html#native-c-consumers)
includes a standalone CMake consumer. Follow the
[coding conventions](https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/style.html)
when contributing. The benchmark guide summarizes measured performance and
limitations, with reproducible commands and detailed historical reports.
