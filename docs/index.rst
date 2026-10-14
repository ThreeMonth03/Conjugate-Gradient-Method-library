Conjugate Gradient Method Library
=================================

``cgpy`` solves nonempty real symmetric positive-definite systems ``A x = b``
with a C++17 core. It supports dense and canonical CSR storage, CG and
Jacobi PCG, reusable workspace, optional OpenMP, an optional dense BLAS
kernel, and an explicit optional CUDA backend. Symmetry and positive
definiteness are caller preconditions.

.. toctree::
   :maxdepth: 2

   install
   build
   api
   cuda
   architecture
   benchmarks

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
For repeated solves, prepare the matrix once and provide one workspace and
output per concurrent call. See :doc:`api` for strict array and ownership
contracts, and :doc:`install` for package installation.
