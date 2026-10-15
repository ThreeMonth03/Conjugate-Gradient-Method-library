Architecture
============

The product solves nonempty real symmetric positive-definite systems with
linear CG and Jacobi PCG. Dense and canonical CSR operators share one solver
engine. Matrix storage, execution policy, iteration state and Python conversion
have separate responsibilities.

Feature layout
--------------

Headers and implementation live together under ``cpp/cg/``. Consumers include
feature headers such as ``<cg/matrix/csr_matrix.hpp>`` or the CPU umbrella
``<cg/cg.hpp>``. CUDA has a separate ``<cg/cuda.hpp>`` umbrella. Installation
exports public headers into ``include/cg/``; private binding helpers stay in
the source tree.

.. list-table:: Source responsibilities
   :header-rows: 1
   :widths: 32 68

   * - Path
     - Responsibility
   * - ``storage/``
     - Owning vectors and borrowed const/mutable vector views.
   * - ``matrix/``
     - Dense matrix ownership/views and validated canonical CSR storage.
   * - ``backend/cpu/``
     - CPU primitives, fused operations, local thread count and dense-kernel
       selection.
   * - ``backend/cuda/``
     - Device/borrowed-stream context and native/cuBLAS operations.
   * - ``preconditioner/``
     - Validate and cache the immutable Jacobi inverse.
   * - ``solver/``
     - Options, statuses, operation counts, reusable workspace and the shared
       CG/PCG engine in ``detail/engine.hpp``.
   * - ``<feature>/cuda/``
     - Device storage, workspace and feature-specific CUDA implementation.
   * - ``<feature>/pymod/``
     - Bind that feature's public types and operations.
   * - ``python/``
     - Extension registration and shared array conversion/lifetime helpers.
   * - ``python/cgpy/``
     - Public ``matrix``, ``storage``, ``solver`` and ``cuda`` packages;
       ``_runtime`` owns native loading and validation support.
   * - ``python/cgbench/``
     - Shared benchmark/test fixtures, independent accuracy gates, balanced
       scheduling and raw measurement artifacts.

The dependency direction is storage/matrix to backend operations to solver
coordination. The preconditioner owns setup state used by the solver. Language
bindings depend on the core; core headers and the ``cg_core`` CMake target do
not depend on Python or pybind11. ``CG_BUILD_PYTHON=OFF`` must build without
locating either dependency. The optional ``cg_cuda`` target owns CUDA SDK
requirements.

The ``cgpy._core`` and optional ``cgpy._cuda`` extension entry points register
feature bindings; they contain no numerical recurrence. Per-feature ``pymod``
code validates layout and ownership, retains borrowed Python owners, guards
workspace concurrency and releases the GIL around native solves. Python
modules import their defining peers rather than private helpers from the
package root. ``cgpy.__init__`` supplies explicit public re-exports.

Workspace and ownership
-----------------------

CPU workspace owns five vectors: residual ``r``, direction ``p``, product
``Ap``, true-residual scratch and preconditioned residual ``z``. Prepare its
size before calling the native solver. A prepared native solve with a prepared
preconditioner does not allocate vector storage; backend/runtime resource
initialization remains separate. Standalone Python calls prepare Jacobi for
that call.

``solve`` normalizes inputs and allocates output/workspace. ``solve_into``
borrows compatible arrays and uses caller-owned output/workspace. ``prepare``
owns a matrix snapshot and optional Jacobi inverse; immutable preparation can
be shared across calls. Preparation fixes thread count and dense-kernel
selection. Keep output/workspace disjoint from inputs and use separate
workspace/output for overlapping solves. Every public solve completes before
returning; borrowed arrays must remain alive and free from conflicting
mutation throughout GIL release.

Numerical policy and kernels
----------------------------

Standard CG and Jacobi PCG share the recurrence. PCG uses ``z=M^-1 r`` and
``r^T z``; convergence always uses the independently recomputed Euclidean true
residual. Periodic residual replacement restarts the direction. Candidate
convergence and final results are verified even when periodic refresh is
disabled. ``algorithm="explicit"`` keeps an explicit-residual reference
formulation under the same stopping contract; Jacobi requires the standard
algorithm.

The backend exposes primitive operations and selected fused operations:
matvec with ``p^T Ap``, solution/residual updates with residual reduction, and
Jacobi application with ``r^T z``. Fusion changes traversal and summation order,
not the CG recurrence. ``fused=False`` retains separate operations for testing
and measurement. Dense fused reductions support odd tails and padded row
strides without fast-math or host-specific compiler flags.

The optional explicit ``dense_kernel="blas"`` choice delegates dense matvec
to a private sequential LP64 OpenBLAS provider. It accepts one CPU thread,
rejects CSR and preserves vector-fusion policy. Native remains the default.
The imported extension reports provider/configuration metadata; see
:doc:`build` for distribution requirements.

Stable norm evaluation avoids false zero from squared-sum underflow. Required
inner products still use ordinary double arithmetic; unusable products return
``non_finite``, even if a reported true residual remains finite. Symmetry and
positive definiteness are caller preconditions. This interface does not claim
arbitrary-scale arithmetic or verify SPD through factorization.

CUDA execution
--------------

CUDA owns explicit device matrices/vectors and borrows a device stream. The
shared solver engine invokes CUDA adapters; it preserves the same recurrence,
true-residual policy and failure statuses. Device workspace owns vectors,
reductions, pinned scalars and a cuBLAS handle on one device/stream.

Calls synchronize before returning and restore the caller's current device.
CPU pointers are never treated as device pointers. External device arrays
retain their owners through GIL release. CUDA Array Interface v3 dependencies
are honored; v2 requires caller-established readiness. Each overlapping solve
requires distinct workspace/output; prepared matrix/Jacobi snapshots can be
shared.

The CPU package has no CUDA runtime dependency. ``cgpy.cuda`` can be imported
with a CPU-only wheel; it loads declared public NVIDIA distributions explicitly
when present. Device transfer methods expose upload/download costs. Resident
prepared/API benchmarks exclude host transfers and record setup separately.
See :doc:`cuda` for exact stream, lifetime, installation and timing contracts.

Validation and change discipline
--------------------------------

GoogleTest checks native ownership, aliasing, kernels, CSR invariants and
numerical breakdown. Pytest checks conversion, strict arrays, snapshots,
GIL/workspace behavior and benchmark contracts. Sphinx imports the real
extension and executes examples. Distribution checks build through an sdist
and exercise a relocated wheel outside the checkout.

Presets isolate serial, OpenMP, BLAS, CUDA, debug and CPU sanitizer builds. CI
uses CPython 3.12 throughout, compiles CUDA on hosted runners and requires a
real device in its explicit self-hosted GPU job. A compile-only result does
not establish GPU correctness or performance. Performance comparisons require
the same machine, inputs and numerical policy; shared CI gates correctness
and artifact collection rather than timing thresholds. Historical benchmark
summaries retain their measured revision and environment in :doc:`benchmarks`;
new raw samples stay under the ignored build directory or in CI artifacts.
