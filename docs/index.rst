Conjugate Gradient Method Library
=================================

``cgpy`` solves real symmetric positive-definite systems ``A x = b`` with CG
and Jacobi PCG. Its C++20 core and CPython 3.12 interface support float64
dense and canonical CSR matrices on CPU or an explicit optional CUDA backend.

First solve
-----------

.. doctest::

   >>> import numpy as np
   >>> import cgpy
   >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
   >>> b = np.array([1.0, 2.0])
   >>> result = cgpy.solve(A, b, rtol=1e-10, preconditioner="jacobi")
   >>> result.status
   'converged'
   >>> np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])
   >>> bool(result.residual_norm <= 1e-10 * np.linalg.norm(b))
   True

The convenience call preserves its inputs and returns an owned solution.
See :doc:`install` for package installation.

API choices
-----------

CPU and CUDA provide corresponding solve, strict and prepared APIs.
Choose based on what should be owned or reused:

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - API
     - Use case
     - Ownership and reuse
   * - ``solve``
     - One solve
     - Fresh output, workspace and any Jacobi setup per call.
   * - ``solve_into``
     - Caller-managed buffers
     - Borrow the matrix and RHS; use caller-owned output and workspace.
   * - ``prepare(...).solve_into``
     - Fixed matrix, repeated RHS
     - Own a matrix snapshot and cache optional Jacobi; use caller-owned output
       and workspace.
   * - ``cuda.Session.solve``
     - CUDA matrices that may change between calls
     - Reuse one fixed-size workspace; borrow each matrix and return a fresh
       output, with Jacobi rebuilt per call.

Each overlapping solve needs its own workspace and output. See :doc:`api`
for strict CPU array contracts and :doc:`cuda` for device data, preparation
and stream lifetimes.

Architecture and limits
-----------------------

* Storage, backend kernels, solver coordination and bindings have separate
  responsibilities; the native core builds without Python.
* RAII owners manage storage and execution resources. Lightweight views borrow
  them, while prepared solvers and sessions expose explicit reuse choices.
* Host and device execution share small numerical policies, including
  true-residual convergence checks, while retaining their own scheduling.

Symmetry and positive definiteness are caller preconditions; the library does
not verify them through factorization. Arithmetic is float64. See
:doc:`architecture` for design decisions and :doc:`benchmarks` for representative
performance, measurement scope and remaining limitations.

Guides
------

.. toctree::
   :maxdepth: 1

   architecture
   api
   cuda
   benchmarks
   install
   build
   style
