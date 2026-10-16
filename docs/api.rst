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
Create CPU prepared solvers with ``cgpy.prepare``; ``cgpy.PreparedSolver``
is the return type and has no public constructor. CUDA follows the same
factory-only contract through ``cuda.prepare`` and ``cuda.prepare_host``.

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
   >>> repeated = cgpy.solve(A, np.array([1.0, 2.0]), preconditioner="jacobi")
   >>> result == repeated
   False
   >>> np.testing.assert_allclose(result.x, repeated.x)

CSR row pointers start at zero, are monotone, and end at ``nnz``. Columns must
be in range, sorted and unique within each row. Construction owns a copy of
the arrays. Jacobi requires a stored finite strictly positive diagonal and a
finite positive inverse. It supports ``algorithm="standard"`` only.

``SolveResult`` fields cannot be rebound, but solution contents remain
writable. Equality compares result object identity; compare numerical
solutions explicitly with a suitable tolerance, as above.

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

Iteration limits and migration
------------------------------

``maxiter`` and ``residual_refresh`` follow the same contract for CPU and CUDA
convenience, strict and prepared solves. They accept Python integers, NumPy
integer scalars and objects implementing ``__index__``. Python booleans retain
their integer meaning of zero or one. Values must be nonnegative and fit the
platform's native ``size_t``; floating-point values are not truncated.
Invalid types, negative values and overflow raise ``ValueError`` naming the
option. Exceptions raised by user-defined ``__index__`` implementations,
other than an invalid integer-conversion ``TypeError``, are propagated.

This intentionally changes CPU prepared solves: callers that previously
caught ``TypeError`` for an invalid iteration limit must catch ``ValueError``.
The accepted numerical options, iteration policy and native C++ ``CgOptions``
representation are unchanged. Convenience calls check limits before allocating
their output/workspace; strict and prepared bindings use the same parser
before entering the solver.
Integer-protocol callbacks are evaluated once per option and complete before
the binding validates and borrows array views; a callback cannot bypass output
writeability checks by changing flags after an earlier descriptor was captured.

CSR conversion owns its native snapshot. Foreign dtype, byte order,
alignment or strides may require an additional normalization buffer; the
native snapshot does not borrow those buffers. NumPy arrays are copied through
the buffer interface without converting each scalar through Python. Canonical
row/column validation remains in the C++ matrix constructor.
