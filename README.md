# Conjugate Gradient Method Library

CPU conjugate gradient solvers for nonempty, real symmetric positive-definite systems `Ax = b`. The C++17 core supports dense and canonical CSR matrices, reusable workspace, fused operations, optional OpenMP, and Jacobi preconditioned CG. Python exposes a convenience API and a strict in-place API. Legacy linear and nonlinear methods remain available through `utils` and `_cgpy`.

## Local development

Use an existing Python 3.12 interpreter and C++17 compiler. Python tools, downloaded C++ dependencies, build products, and caches stay under `.venv/`, `build/`, and `.cache/` in this checkout. These commands require no system package upgrades or shell configuration changes.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/test.lock
.venv/bin/python -m pip install --cache-dir .cache/pip --no-deps --no-build-isolation -e .
make test
```

`make pytest` runs Python tests, `make gtest` runs native GoogleTest cases, and `make lint` checks Python lint/formatting and C++ formatting. `make gtest PRESET=cpu-asan` runs the native tests under AddressSanitizer and UndefinedBehaviorSanitizer. `make wheel` builds a wheel under `dist/`; `make demo` runs the examples.

The default build uses one CPU thread. `make test PRESET=cpu-openmp` enables OpenMP. For an editable installation that supports `num_threads > 1`, add `--config-settings=cmake.define.CG_ENABLE_OPENMP=ON` to the editable install command. Presets build into separate directories; selecting an OpenMP preset for a Make target does not change an already installed wheel.

## Python example

```python
import numpy as np
import cgpy

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
result = cgpy.solve(A, b, rtol=1e-10, preconditioner="jacobi")
assert result.converged
print(result.x, result.iterations, result.residual_norm)
```

`solve` preserves the inputs and owns its returned solution. `prepare(A, preconditioner="jacobi")` snapshots the matrix and caches its preconditioner for repeated solves. Create a workspace with `prepared.create_workspace()`, then call `prepared.solve_into(b, x, workspace=workspace)`. The supplied `x` is both the initial guess and the output; reset it before a repeated comparison. Each concurrent solve needs a distinct workspace and output.

`cgpy.CSRMatrix(indptr, indices, data, shape)` is accepted by `solve`, `prepare`, and `solve_into`. CSR rows must have sorted, unique column indices. Jacobi requires a finite, strictly positive diagonal and is supported with `algorithm="standard"`.

The stopping rule is `||b - A @ x||2 <= max(atol, rtol * ||b||2)`. Reported residuals are independently recomputed from the final solution. Check `converged` or `status` before using the answer. Symmetry and positive definiteness are caller preconditions; the solver does not perform a full matrix factorization to verify them.

See [Python and C++ interfaces](python/README.md) for options, ownership, and failure statuses.
See [architecture](docs/architecture.md) for storage, workspace, backend, and fusion boundaries.

## Reproducible comparisons

Install the separate CPU benchmark dependencies into the same local environment:

```bash
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/benchmark.lock
make pyprof
make pyprof PRESET=cpu-openmp PYPROF_THREADS=4
make pyprof PYPROF_SCOPE=api
make pyprof PYPROF_SUITE=cpu-dense-f64-v1
make pyprof PYPROF_ARGS='--warmups 3 --seed 20261015'
```

The default sparse solver suite compares native CG and Jacobi PCG, SciPy CG and PCG, and SuperLU factorization plus solve versus factorization reuse. The dense suite compares CG implementations using NumPy and eager CPU PyTorch. Direct and iterative methods have different setup costs and iteration counts; retain each result's scope and solver labels when comparing them. Native and official SciPy iterative policies are labeled separately; matching-policy reference adapters are available through the CLI.

The CLI and pytest use the same cases and validation fixtures. Candidates must satisfy an independently checked true residual before timing. Timed rounds use complete Williams cycles to balance execution order. JSON artifacts include raw timings, schedules, input hashes, solver options, and environment/build metadata. `PYPROF_OUTPUT` selects the artifact path; `PYPROF_ARGS` forwards additional CLI arguments. See [benchmark details](benchmarks/README.md) for scopes and replay.

Performance depends on matrix structure, conditioning, hardware, thread count, and setup reuse. Compare on the same machine with the same tolerances and initial guesses; shared-runner CI checks correctness rather than enforcing speed thresholds.

An optional [CPU Docker environment](contrib/docker/README.md) keeps the compiler and dependencies inside a container.
