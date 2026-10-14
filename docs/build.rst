Build, test and document
========================

Run from the repository root with an existing Python interpreter and C++17
compiler. Install the local test tools before using Make:

.. code-block:: bash

   python3 -m venv .venv
   .venv/bin/python -m pip install --cache-dir .cache/pip -r requirements/test.lock
   make test
   make test PRESET=cpu-openmp
   make gtest PRESET=cpu-asan
   make lint
   make demo

``make test`` runs GoogleTest and pytest through CTest. ``make gtest`` and
``make pytest`` select their respective layers. The sanitizer preset runs
native address and undefined-behavior checks. ``make lint`` checks Python lint
and formatting plus C++ formatting. Build directories are isolated by preset
under ``build/``. ``JOBS`` controls build parallelism; ``PYTHON``, ``BUILD_DIR``
and ``CMAKE_ARGS`` allow local toolchain overrides.

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
environment outside the source tree and checks dense/CSR APIs and native
imports. Native and BLAS wheels have the same package version and wheel tag;
use separate output directories to retain both artifacts. Runtime-only
installs need NumPy; the legacy derivative dependency stays optional.

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
