Build, test and document
========================

Run from the repository root with CPython 3.12.x and a C++20 compiler. Install the local test tools before using Make:

.. code-block:: bash

   python3.12 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/test.lock
   make test
   make test PRESET=cpu-openmp
   make gtest PRESET=cpu-asan
   make lint
   make typecheck
   make test-native
   make demo

``make test`` runs GoogleTest and pytest through CTest. ``make gtest`` and
``make pytest`` select their respective layers. The sanitizer preset runs
native address and undefined-behavior checks. Python tests live under
``tests/python/``; examples live under ``examples/``. ``make lint`` checks Python lint
and formatting plus C++ formatting. Build directories are isolated by preset
under ``build/``. ``JOBS`` controls build parallelism; ``PYTHON``, ``BUILD_DIR``
and ``CMAKE_ARGS`` allow local toolchain overrides.

``make typecheck`` checks representative public CPU and CUDA calls with the
pinned mypy development tool. It copies the staged package into an isolated
consumer directory, verifies that importing the CUDA façade does not load the
optional extension, and checks return types and rejected API usage. No GPU or
Torch/CuPy installation is needed. The checks live under ``tests/typing/``;
the benchmark package is outside their scope.

``make test-native`` builds the selected preset with Python and internal tests
disabled in ``build/<preset>-native``. It installs the development component,
moves the installation prefix, and configures an independent C++ consumer
outside the checkout. The consumer checks dense/CSR CG and PCG, reusable
workspaces, true residuals and basic operation counts. Native, OpenMP and BLAS
CI configurations run these checks. CUDA CI also compiles/links the optional
consumer and reports device-dependent skips; the GPU job uses
``NATIVE_ARGS=--require-cuda``. ``NATIVE_BUILD_DIR`` overrides its build path.
See :doc:`install` for direct CMake commands and imported targets.

No command above upgrades system packages, installs with ``pip --user``, or
changes shell configuration. CMake downloads pinned GoogleTest into its local
build dependency directory. The optional BLAS dependency also belongs in the
project's local dependency directory.

Optional dense BLAS
-------------------

The BLAS kernel replaces dense matrix-vector products. Vector updates and
reductions retain the selected fused/separate CPU policy. ``dense_kernel`` is
independent of ``algorithm``, ``fused`` and Jacobi preconditioning. The first
implementation accepts only ``num_threads=1``; native OpenMP remains available
for multiple threads. CSR operators require ``dense_kernel="native"``.

Build the optional mode explicitly:

.. code-block:: bash

   make test PRESET=cpu-blas
   make docs PRESET=cpu-blas
   make doctest PRESET=cpu-blas

The Linux native build needs GNU or LLVM ``nm`` and ``objcopy`` in the
existing toolchain. CMake downloads pinned OpenBLAS 0.3.30 and builds a private
sequential LP64 double-precision BLAS provider under ``build/_deps``. It needs
no Fortran compiler or LAPACK build. Initial dependency downloads require
network access. Wheel builds statically include the provider and use private
symbol names, so installed wheels do not depend on that source/build directory.

On x86-64, a BLAS-enabled binary or wheel requires a CPU supporting SSE3 and
SSSE3 (the CORE2 kernel baseline). The provider dispatches between CORE2 and
Haswell kernels on supported CPUs. This requirement applies to the entire
BLAS-enabled extension, including import; selecting the native kernel does
not remove it. Other Linux architectures use the provider's native build
target and require a compatible CPU when relocating the wheel. These wheels
are local platform builds; no manylinux portability is claimed.

Inspect ``cgpy.blas_enabled()`` and ``cgpy.build_info()`` in the process that
will run the solve. See :doc:`api` for the runtime contract.

.. doctest:: blas-enabled
   :skipif: not cgpy.blas_enabled()

   >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
   >>> b = np.array([1.0, 2.0])
   >>> result = cgpy.solve(A, b, dense_kernel="blas", num_threads=1)
   >>> result.converged
   True
   >>> np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])

A native-only build rejects the unavailable choice explicitly:

.. doctest:: blas-unavailable
   :skipif: cgpy.blas_enabled()

   >>> try:
   ...     cgpy.solve(np.eye(2), np.ones(2), dense_kernel="blas")
   ... except ValueError:
   ...     rejected = True
   ... else:
   ...     rejected = False
   >>> rejected
   True

Wheel and source distributions
------------------------------

.. code-block:: bash

   make wheel
   make sdist
   make dist
   .venv/bin/python -m pip wheel --cache-dir .cache/pip --wheel-dir build/sdist-wheel dist/cgpy-*.tar.gz
   make test-dist DIST_BLAS=OFF DIST_DIR=dist/native
   make test-dist DIST_BLAS=ON DIST_DIR=dist/blas

``make wheel`` builds a wheel with isolated backend dependencies, ``make
sdist`` builds a source distribution, and ``make dist`` builds both through
the source-distribution path. The pip command builds an sdist with isolation.
``make test-dist`` additionally installs that wheel into a separate temporary
environment outside the source tree and checks dense/CSR APIs, native
imports and public types against that installed environment. Both native
stubs and the ``py.typed`` marker must survive the sdist-to-wheel path. The
type checker runs in the development environment; it is not a runtime
dependency of the installed wheel.
Native and BLAS wheels have the same package version and wheel tag;
use separate output directories to retain both artifacts. Runtime-only
CPU installs need NumPy; Python 3.12 is required for every package mode.

CUDA builds use ``PRESET=cuda-release`` and ``DIST_CUDA=ON`` explicitly, with
their own compatible local toolkit and runtime dependencies. See :doc:`cuda`
for complete commands and architecture/host-compiler overrides. CPU wheels
retain ``CG_ENABLE_CUDA=OFF``. Distribution checks require CUDA sources in the
sdist, inspect installed ELF dependencies/RPATH, and execute GPU smoke solves
when a usable device exists; GPUless installation checks report that skip.

Sphinx documentation
--------------------

.. code-block:: bash

   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/docs.lock
   make docs
   make doctest

The HTML entry point is ``build/docs/html/index.html``. The doctest builder
executes native dense, CSR and prepared-solver examples; a BLAS build also
executes the BLAS example above. Availability-specific groups use explicit
runtime skip conditions; native-only builds test unavailable-mode rejection.
Both builders treat warnings as
errors. Autodoc imports the selected real extension through the same
``PYTHONPATH`` as the other Make targets, and Napoleon renders NumPy-style
docstrings. No imports are mocked. Once dependencies and the extension are
available, Sphinx requires no network access: the configuration uses no remote
theme assets or intersphinx inventories.

GitHub Pages publishes the HTML built from ``main`` at
`the documentation site <https://ThreeMonth03.github.io/Conjugate-Gradient-Method-library/>`_.
The ``Documentation`` workflow builds the real CPU extension, checks Sphinx
warnings and executes doctests on every pull request. After a merge to
``main``, it uploads the HTML and deploys it to the ``github-pages``
environment. The repository's Pages source is GitHub Actions; generated HTML
stays out of Git. BLAS and CUDA documentation examples are additionally
checked in their respective solver CI jobs.

Python version and dependency policy
------------------------------------

Python support is intentionally limited to CPython 3.12.x. CI varies the
execution/build configuration rather than testing unrelated interpreter
versions. Use a current 3.12 patch release. Native-only builds use
``CG_BUILD_PYTHON=OFF`` and have no Python requirement.

The requirement files are explicit pinned pip inputs, maintained by review;
there is no automatic lock generator. See
`dependency maintenance <https://github.com/ThreeMonth03/Conjugate-Gradient-Method-library/blob/main/requirements/README.md>`_
for clean-environment installation and update checks.
