Architecture
============

The core separates storage, execution, solver state and language bindings.
Dense and canonical CSR operators share the CG/PCG implementation. Matrix
storage does not select a solver or change process-wide thread settings.

* ``storage.hpp`` defines borrowed const/mutable views, owning vectors/dense
  matrices, and a workspace with residual, direction, product, true-residual
  scratch and preconditioned-residual vectors.
* ``csr.hpp`` owns validated canonical row pointers, columns and values.
* ``backend.hpp`` supplies synchronous primitive and fused CPU operations,
  local native thread settings, and dense-kernel selection.
* ``preconditioner.hpp`` validates and owns the immutable Jacobi inverse.
* ``solver.hpp`` defines CPU solver entry points, statuses and operation
  counts; ``detail/solver_engine.hpp`` shares the recurrence, stopping rule and
  refresh/restart policy between CPU and CUDA.
* ``cuda.hpp`` defines device ownership/views, borrowed stream context,
  workspace and explicit native/cuBLAS kernels. CUDA SDK details are private
  to the optional build.
* The binding validates arrays, retains owners, guards workspace concurrency
  and releases the GIL around the native solve.
* ``cgpy`` provides conversion and result ownership. ``cgbench`` provides
  common test/CLI fixtures, independent accuracy gates and raw timing records.

Prepared solves reuse matrix and Jacobi setup. The core uses caller-owned
workspace vectors; the backend runtime may initialize its own resources.
Native fusion combines traversal where useful without changing the chosen CG
recurrence. The dense BLAS path delegates matrix-vector multiplication while
retaining the CPU vector policy and true-residual verification. BLAS is an
explicit optional build/runtime choice, with its first interface restricted to
one thread and dense storage.

Standard CG and PCG periodically replace the residual and restart direction.
Candidate convergence and final results are verified even with periodic
refresh disabled. Stable norm evaluation avoids false zero from squared-sum
underflow; required inner products still use ordinary double arithmetic.

The optional CUDA target uses explicit device storage/views and a borrowed
stream context. Workspace owns vectors, reductions, pinned scalars and a
cuBLAS handle; the shared solver engine invokes device operations and retains
the same true-residual policy. Public calls synchronize before returning and
restore the caller's current device. CPU pointers are never reinterpreted as
device pointers. Device prepared/API benchmarks separate host transfer/setup
from completed resident solve timing. See :doc:`cuda` for ownership, stream,
runtime installation and interoperability contracts.
