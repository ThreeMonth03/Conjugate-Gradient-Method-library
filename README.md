# Conjugate Gradient Method Library

A C++17 library with Python bindings for real symmetric positive-definite
systems `Ax = b`. The modern API provides dense and canonical CSR CG, Jacobi
PCG, reusable workspaces and prepared solves. CPU execution supports native
kernels, optional OpenMP and optional dense BLAS. CUDA is an explicit optional
backend with native kernels and dense cuBLAS.

## Quick start

Use **CPython 3.12.x** and a C++17 compiler. From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip .
.venv/bin/python -c 'import cgpy; print(cgpy.build_info())'
```

Pip builds in an isolated environment. A compatible built wheel needs no
compiler; the default CPU package requires NumPy at runtime.

```python
import numpy as np
import cgpy

A = np.array([[4.0, 1.0], [1.0, 3.0]], dtype="float64")
b = np.array([1.0, 2.0], dtype="float64")
result = cgpy.solve(A, b, preconditioner="jacobi", rtol=1e-10)
assert result.converged
np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])
```

`solve` preserves inputs and owns its output. `prepare(A)` snapshots the
matrix and caches setup; `prepared.solve_into(b, x, workspace=workspace)`
reuses caller-owned output and workspace. Each concurrent solve needs its own
workspace and output. `cgpy.CSRMatrix` stores canonical sparse inputs without
converting them to dense matrices.

Convergence requires the independently recomputed true residual to satisfy
`||b - A @ x||2 <= max(atol, rtol * ||b||2)`. Inspect `converged` or `status`
before using a solution. Symmetry and positive definiteness are caller
preconditions. See [interfaces and ownership](python/README.md).

## Develop and validate

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/test.lock
.venv/bin/python -m pip install --cache-dir .cache/pip --no-deps --no-build-isolation -e .
make test
make lint
make gtest PRESET=cpu-asan
make demo
```

`make test` runs native GoogleTest and Python pytest; `make gtest` and
`make pytest` select one layer. Presets keep independent builds under
`build/`. Use `PRESET=cpu-openmp` or `PRESET=cpu-blas` for the optional CPU
modes. Preset selection does not alter an already installed package; inspect
`cgpy.build_info()` in the process that will perform the solve.

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/docs.lock
make docs doctest
make test-dist
```

Sphinx imports the actual extension, treats warnings as errors and executes
examples. `make test-dist` builds a wheel through an sdist, installs it outside
the checkout and checks the relocated package. See [build and distribution
checks](docs/build.rst), [dependency pin maintenance](requirements/README.md)
and the optional [CPU Docker environment](contrib/docker/README.md).

## Structure and design

```text
cpp/cg/
  storage/          # Ownership and borrowed vector views
  matrix/           # Dense and canonical CSR representations
  backend/          # CPU and CUDA numerical operations
  preconditioner/   # Jacobi setup and application
  solver/           # Shared CG/PCG engine, status and workspace
  python/           # Extension entry points and shared conversion boundary
python/
  cgpy/             # matrix, storage, solver, _runtime and cuda packages
  cgbench/          # Accuracy gates, benchmark adapters and artifacts
tests/
  cpp/              # Core invariants and native numerical behavior
  python/           # API, matrix, solver, CUDA and benchmark contracts
examples/           # Dense, sparse and repeated-solve entry points
```

Each C++ feature keeps declarations and implementation together; its Python
binding lives in `pymod/`. The core builds without Python or pybind11. Matrix
storage does not choose algorithms or thread policy. CPU and CUDA adapters
share the solver engine. The [architecture guide](docs/architecture.rst)
describes dependency direction, ownership and numerical policy; [STYLE.md](STYLE.md)
records the SOLVCON-inspired conventions enforced by the formatters.

The supported product is linear CG/PCG. The earlier `_cgpy`, `utils`, matrix
subclasses and nonlinear APIs are retired. `algorithm="explicit"` remains a
modern reference path for comparison under the shared convergence contract.

## Performance evidence

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
make pyprof PYPROF_SUITE=smoke-sparse-solvers
make pyprof PYPROF_SUITE=cpu-dense-f64-v1
make pyprof PYPROF_SCOPE=api
```

Candidates pass an independent true-residual gate before timing. Artifacts
retain input hashes, solver options, raw timings, balanced schedules and
hardware/build metadata. Prepared and API scopes separate setup reuse from
per-call allocation. Shared CI checks correctness; performance comparisons
require the same machine, inputs, tolerances and thread policy.

[Benchmark contracts](benchmarks/README.md) explain reproduction and replay.
Recorded [CPU](benchmarks/results.md), [BLAS](benchmarks/blas-results.md) and
[CUDA](benchmarks/cuda-results.md) measurements retain their original artifacts
and limitations; they are historical evidence, not measurements of every
subsequent source revision.

The [refactor validation report](docs/refactor_validation.rst) records the
current numerical, installation and native CPU performance checks, including
all raw measurements and the limits of the comparisons.

## CUDA

CUDA uses a separate environment, an existing NVIDIA driver and a compatible
host compiler. The pinned Linux x86-64 bootstrap provides Toolkit 12.8 inside
the checkout. The CPU wheel has no NVIDIA runtime dependency.

```bash
python3.12 -m venv .venv-cuda
.venv-cuda/bin/python -m pip install --cache-dir .cache/pip \
    -r requirements/test.lock -r requirements/cuda.lock
.venv-cuda/bin/python scripts/setup_cuda.py --prefix .deps/cuda-12.8
CG_REQUIRE_CUDA=1 make test PRESET=cuda-release PYTHON="$PWD/.venv-cuda/bin/python"
```

Import `from cgpy import cuda`, explicitly upload matrices/vectors, then select
native or cuBLAS kernels. Calls complete on the selected stream before
returning. See the [CUDA guide](docs/cuda.rst) for runtime installation,
architecture overrides, interoperability and resident benchmark scope.
