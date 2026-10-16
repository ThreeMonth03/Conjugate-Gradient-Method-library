# Dependency pins

The project supports CPython 3.12.x. `.python-version` selects the minor
version; use a current 3.12 patch release. Native-only C++ builds do not require
Python. The package metadata, CMake Python check, Ruff target, Docker image
and CI must agree on the Python version policy.

The NumPy runtime floor is 1.26, the first series with Python 3.12 support
([upstream release notes](https://numpy.org/doc/stable/release/1.26.0-notes.html)).

These `.lock` files are checked-in pip requirements with exact version pins
and platform markers. They are maintained explicitly; there is no lock-file
generator or hash-verified cross-platform resolution in this repository.

| File | Purpose |
| --- | --- |
| `dev.lock` | Build tools, formatters, pytest and mypy |
| `test.lock` | Development tools plus SciPy and thread-pool inspection |
| `docs.lock` | Development tools plus Sphinx and its dependencies |
| `packaging.lock` | Build and type-check tools for distribution validation |
| `benchmark.lock` | CPU comparisons, including CPU-only PyTorch |
| `cuda.lock` | Public NVIDIA runtime and cuBLAS packages |
| `cuda-benchmark.lock` | Tests plus CUDA PyTorch and its runtime dependencies |

Install with the chosen environment's interpreter:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/test.lock
.venv/bin/python -m pip check
```

CPU and CUDA PyTorch distributions belong in separate environments. The CUDA
locks retain Linux/platform markers because the pinned CUDA bootstrap targets
Linux x86-64. A requirements installation is not evidence that a CUDA device
was exercised.

To update pins, use a fresh CPython 3.12 environment. Resolve the intended
package updates there, inspect their transitive requirements with
`python -m pip inspect` and `python -m pip freeze`, and edit the corresponding
pin files. Freeze output is evidence for review, not a replacement for the
separate CPU/CUDA requirement sets. Keep shared build pins aligned with
`pyproject.toml`.

Reinstall each affected requirement set into a clean environment and run
`python -m pip check`. Validate native/Python tests, formatting, documentation
and wheel installation as applicable; Linux CI covers the optional BLAS and
CUDA builds. Benchmark dependency updates require regenerated input-hash and
numerical checks before comparing new timing artifacts with recorded results.
Keep original historical artifacts unchanged.
