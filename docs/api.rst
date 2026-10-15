Python API
==========

The CPU calls are synchronous. Numerical breakdown and exhausted iterations
return a ``SolveInfo`` status; invalid dimensions, layouts, options and Jacobi
setup raise an exception. The stopping rule uses the independently evaluated
true residual:

.. code-block:: text

   ||b - A @ x||2 <= max(atol, rtol * ||b||2)

Symmetry and positive definiteness are caller preconditions. ``non_finite``
also reports required inner products outside the usable double range, even
when the final reported norm is finite.

Convenience and strict functions
--------------------------------

.. autofunction:: cgpy.solve

.. autofunction:: cgpy.prepare

.. autofunction:: cgpy.solve_into

``solve`` normalizes real array-like inputs and owns its output. The strict
``solve_into`` function and prepared methods require aligned C-contiguous
float64 NumPy vectors, with a writable output. Dense strict inputs must also
already have that dtype and layout. Output must be disjoint from the matrix
and right-hand side. Keep borrowed arrays alive and free from conflicting
mutation until the call returns.

``dense_kernel="native"`` is the default. The optional ``"blas"`` choice is
dense-only and accepts ``num_threads=1``. ``fused`` independently controls the
remaining CPU vector operations. A prepared solver fixes its matrix snapshot,
thread count, preconditioner and dense-kernel choice when it is constructed.
Dense and CSR inputs return the same ``PreparedSolver`` type. ``CsrMatrix``
is the public spelling in both CPU and CUDA APIs.

Repeated solves
---------------

.. doctest::

   >>> A = np.array([[4.0, 1.0], [1.0, 3.0]])
   >>> b = np.array([1.0, 2.0])
   >>> plan = cgpy.prepare(A, preconditioner="jacobi")
   >>> workspace = plan.create_workspace()
   >>> x = np.zeros_like(b)
   >>> info = plan.solve_into(b, x, workspace=workspace, rtol=1e-10)
   >>> info.converged
   True
   >>> np.testing.assert_allclose(x, [1 / 11, 7 / 11])
   >>> x.fill(0.0)
   >>> plan.solve_into(b, x, workspace=workspace, rtol=1e-10).converged
   True

Each overlapping solve needs a distinct workspace and output. Prepared matrix
and preconditioner snapshots can be shared. Long native computation releases
the GIL; this does not make concurrent mutation of borrowed NumPy arrays safe.

.. autoclass:: cgpy.PreparedSolver
   :members:

.. autoclass:: cgpy.Workspace
   :members:

CSR and results
---------------

.. autoclass:: cgpy.CsrMatrix
   :members:

.. doctest::

   >>> A = cgpy.CsrMatrix([0, 2, 4], [0, 1, 0, 1], [4.0, 1.0, 1.0, 3.0], (2, 2))
   >>> A.shape, A.nnz
   ((2, 2), 4)
   >>> result = cgpy.solve(A, np.array([1.0, 2.0]), preconditioner="jacobi")
   >>> result.converged
   True
   >>> np.testing.assert_allclose(result.x, [1 / 11, 7 / 11])

CSR row pointers start at zero, are monotone, and end at ``nnz``. Columns must
be in range, sorted and unique within each row. Construction owns a copy of
the arrays. Jacobi requires a stored finite strictly positive diagonal and a
finite positive inverse. It supports ``algorithm="standard"`` only.

.. autoclass:: cgpy.SolveResult
   :members:

.. autoclass:: cgpy.SolveInfo
   :members:

Build capabilities
------------------

.. autofunction:: cgpy.openmp_enabled

.. autofunction:: cgpy.blas_enabled

.. autofunction:: cgpy.build_info

These report the extension actually imported by this Python process. They do
not describe a different CMake build or a wheel installed elsewhere.
