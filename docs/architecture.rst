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
exports public headers into ``include/cg/``. Private implementation helpers
live in ``detail/``; binding helpers stay in the source tree.

.. list-table:: Source responsibilities
   :header-rows: 1
   :widths: 32 68

   * - Path
     - Responsibility
   * - ``runtime/detail/``
     - Shared overflow-checked size arithmetic, independent of CUDA.
   * - ``runtime/cuda/``
     - Device selection, borrowed stream context and low-level runtime helpers.
   * - ``storage/``
     - Standard host containers/views and CUDA allocation, vectors and fill.
   * - ``matrix/``
     - Dense matrix ownership/views, validated canonical CSR storage and
       shared strided-storage extent checks.
   * - ``backend/cpu/``
     - CPU primitives, fused operations, local thread count and dense-kernel
       selection.
   * - ``backend/cuda/``
     - Native/cuBLAS operations and private execution resources: reductions,
       inline host scalar and optional cuBLAS handle/scratch.
   * - ``preconditioner/``
     - Validate and cache the immutable Jacobi inverse.
   * - ``solver/``
     - Options, result metadata, reusable workspace, owning ``PreparedSolver``
       and the shared CG/PCG engine in ``detail/engine.hpp``.
   * - ``<feature>/cuda/``
     - Device storage, workspace and feature-specific CUDA implementation.
   * - ``<feature>/pymod/``
     - Register that feature's CPU/CUDA APIs in ``cpu.cpp``/``cuda.cpp``;
       parse Python configuration at the language boundary.
   * - ``python/``
     - Extension registration and shared array conversion/lifetime helpers.
   * - ``python/cgpy/``
     - Public ``matrix``, ``storage``, ``solver`` and ``cuda`` packages;
       ``_runtime`` loads native modules, ``_interop`` normalizes host
       representations, and ``solver/_validation.py`` checks solve policy.
   * - ``python/cgbench/``
     - ``adapters/`` invoke implementations; ``reference/`` owns an independent
       recurrence with a small array-operations protocol. ``execution.py``
       defines their result contract and explicit candidate factory.

Dependencies point from solver coordination to preconditioner/backend, then
matrix/storage and runtime. Lower layers never include solver state. The
preconditioner owns setup state used by the solver. Language
bindings depend on the core; core headers and the ``cg_core`` CMake target do
not depend on Python or pybind11. ``CG_BUILD_PYTHON=OFF`` must build without
locating either dependency. The optional ``cg_cuda`` target owns CUDA SDK
requirements.

The ``cgpy._core`` and optional ``cgpy._cuda`` extension entry points register
feature bindings; they contain no numerical recurrence. Per-feature ``pymod``
code validates layout and ownership, retains borrowed Python owners, guards
workspace concurrency and releases the GIL around native solves. Prepared
matrix ownership and Jacobi setup belong to native ``solver/`` classes,
so C++ consumers use them without importing Python. Python
modules import their defining peers rather than private helpers from the
package root. ``cgpy.__init__`` supplies explicit public re-exports.

Workspace and ownership
-----------------------

Views borrow storage without owning or extending its lifetime. CPU vector
views use ``std::span``. CPU dense and CUDA vector/dense/CSR views keep their
pointer, extent and layout/device metadata private and expose read-only
accessors such as ``data()``, ``size()``, ``rows()`` and ``device()``. Replacing
a view replaces its whole descriptor; callers cannot edit individual fields.
CUDA views remain trivially copyable descriptors with host/device accessors
and no allocation or synchronization. Device pointer inspection and stream
readiness checks remain at operation and foreign-array boundaries. Metadata
encapsulation cannot establish the underlying allocation's extent or lifetime.

CPU ``CsrMatrix`` owns its three vectors and exposes their contents through
read-only spans: ``indptr()`` and ``indices()`` return
``std::span<const std::size_t>``; ``values()`` returns ``ConstVectorView``
(``std::span<const double>``). The getters borrow without allocation and reject
calls on rvalues. Keep the backing storage alive; assignment or destruction
can invalidate a span. Reacquire views after moving or swapping the matrix.
Copy explicitly when an independent snapshot is needed:

.. code-block:: cpp

   cg::CsrMatrix matrix(2, 2, {0, 2, 4}, {0, 1, 0, 1}, {4., 1., 1., 3.});
   auto values = matrix.values();
   std::vector<double> snapshot(values.begin(), values.end());

CPU and CUDA allocation/layout checks share the same checked multiplication
and strided-storage extent functions. Backend-specific pointer and device
validation stays at its existing boundary.

CPU workspace owns five vectors: residual ``r``, direction ``p``, product
``Ap``, true-residual scratch and preconditioned residual ``z``. Prepare its
size before calling the native solver. It is noncopyable and movable; a
moved-from workspace is empty and reusable. ``capacity()`` is the minimum
capacity of all five buffers. ``prepare()`` commits growth only after every
allocation succeeds and preserves existing state on failure. A prepared
native solve with a prepared
preconditioner does not allocate vector storage; backend/runtime resource
initialization remains separate. Standalone Python calls prepare Jacobi for
that call.

``solve`` normalizes inputs and allocates output/workspace. ``solve_into``
borrows compatible arrays and uses caller-owned output/workspace. ``prepare``
owns a matrix snapshot and optional Jacobi inverse; immutable preparation can
be shared across calls. Dense and CSR preparation return the same concrete
``PreparedSolver`` type. ``CsrMatrix`` has the same spelling in C++ and Python.
Preparation fixes thread count and dense-kernel
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
and measurement. Native CPU dense matvec uses the same optimized paired-row
kernel with either fusion setting. Separate reductions can change
floating-point rounding and iteration counts; both policies retain
true-residual verification. Dense fused reductions support odd tails and
padded row strides without fast-math or host-specific compiler flags.

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

CUDA owns explicit device matrices/vectors and borrows a device stream. Host
execution uses the shared solver engine with CUDA adapters. Explicit device
execution keeps recurrence scalars on the GPU and reads completed status
packets; it follows the same recurrence, true-residual policy and failure
statuses. Workspace construction fixes host/device execution independently of
dense-kernel and vector-fusion choices. Device workspace composes solver
vectors and private backend execution resources on one device/stream. A
solver-local RAII guard acquires the workspace and lends those resources to
the backend; the backend never sees solver state. Only resources prepared
for the cuBLAS kernel own a handle and 4 MiB scratch allocation.
The optional cuBLAS resource bundle is held directly in the execution state;
its device allocations retain their own RAII owners. A prepared solver derives
its preconditioner kind from the optional cached Jacobi state.

Python ``cuda.Session`` owns one existing CUDA workspace with fixed size,
device, stream, kernel and execution mode. It borrows matrix inputs for each
call and returns a fresh owning output, with Jacobi rebuilt per call. It
retains no operator/preconditioner cache and permits one active call, rejecting
overlap. See :doc:`cuda` for the public lifetime contract.

CUDA CSR storage directly composes two RAII-owned ``int64`` index arrays and
a CUDA ``Vector`` for values. It derives ``nnz`` and device identity from the
values owner. The matrix is noncopyable and has nonthrowing moves; a moved-from
matrix has zero dimensions, no storage and device ``-1``. Workspace keeps a
private implementation to hide CUDA library execution resources.

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
and exercise a relocated wheel outside the checkout. A small architecture
test checks native feature include directions, excludes Python from the core,
and prevents the product package from importing benchmark code.

Presets isolate serial, OpenMP, BLAS, CUDA, debug and CPU sanitizer builds. CI
uses CPython 3.12 throughout, compiles CUDA on hosted runners and requires a
real device in its explicit self-hosted GPU job. A compile-only result does
not establish GPU correctness or performance. Performance comparisons require
the same machine, inputs and numerical policy; shared CI gates correctness
and artifact collection rather than timing thresholds. Historical benchmark
summaries retain their measured revision and environment in :doc:`benchmarks`;
new raw samples stay under the ignored build directory or in CI artifacts.
