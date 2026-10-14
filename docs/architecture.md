# CPU solver architecture

The core separates numerical storage, execution, solver state, and language bindings. Dense and CSR operators use the same CG/PCG implementation; neither matrix class owns a solver or a thread policy.

```mermaid
flowchart TD
    Python[Python convenience / strict APIs] --> Binding[pybind11 validation and lifetime boundary]
    Legacy[Legacy linear wrapper] --> Solver[Shared CG / PCG algorithm]
    Binding --> Solver
    Solver --> Backend[CpuBackend operations]
    Solver --> Workspace[Caller-owned Workspace]
    Solver --> Jacobi[Optional immutable Jacobi inverse]
    Backend --> Dense[DenseMatrixView / owning DenseMatrix]
    Backend --> CSR[Owning canonical CsrMatrix]
```

| Layer | Responsibility |
| --- | --- |
| `include/cg/storage.hpp` | Borrowed const/mutable views, owning dense/vector storage, reusable workspace. Views retain no Python owner. |
| `include/cg/csr.hpp` | Owning, validated CSR with sorted unique columns, float64 values, and `size_t` indices. Storage is immutable after construction. |
| `include/cg/backend.hpp` | Synchronous CPU operations and per-instance thread/fusion settings. No process-wide thread changes. |
| `include/cg/preconditioner.hpp` | Validate and cache a positive Jacobi inverse; apply it through backend operations. |
| `include/cg/solver.hpp` | Shared stopping rules, recurrence, refresh/restart, failure status, and operation counts. |
| `bindings/cpu.cpp` | Strict array validation, prepared snapshots, owner lifetimes, GIL release, and a concurrent-use guard on Python workspace. |
| `python/cgpy` | Convenience conversion and result ownership; strict and prepared APIs expose reuse explicitly. |
| `python/cgbench` | Shared pytest fixtures, independent accuracy checks, balanced scheduling, and raw artifacts. |

## Workspace and ownership

Workspace owns five vectors: residual `r`, direction `p`, product `Ap`, true-residual scratch, and preconditioned residual `z`. Prepare its size before calling the native solver. The native core with a prepared preconditioner does not allocate vector storage. Standalone Python calls with Jacobi construct its inverse diagonal for that call. Keep the output and workspace disjoint from all inputs; use one workspace and output per concurrent solve.

`solve` normalizes array-like inputs and allocates the result and workspace. `solve_into` borrows compatible arrays and uses caller-owned output/workspace. `prepare` owns a matrix snapshot and an optional Jacobi inverse, so repeated solves can share those immutable objects. Each API call is synchronous; borrowed arrays must remain alive and free from conflicting mutation until it returns. The binding releases the GIL only around the native solve.

## Operations and numerical policy

The backend exposes primitive operations and selected fused operations: matvec with `pᵀAp`, solution/residual update with residual reduction, and Jacobi application with `rᵀz`. `fused=False` keeps a separate-operation reference path. Fusion changes data traversal and summation order; it does not select the CG algorithm.

Dense fused row reductions use independent accumulators to permit vectorization without fast-math or host-specific compiler flags. Odd tails and padded row strides are supported; tests compare against long-double sums and retain non-finite multiplication behavior. `make pyprof PYPROF_ARGS='--no-fused'` selects the native reference path for measurement.

Standard CG and Jacobi PCG share the recurrence. PCG uses `z=M⁻¹r` and `rᵀz`, while convergence always uses the Euclidean true residual. Periodic replacement restarts the direction from the recomputed residual (or its preconditioned value). Candidate convergence and final results are verified even when periodic refresh is disabled. The explicit legacy algorithm remains selectable and does not support Jacobi.

Stable norm evaluation prevents false zero residuals from squared-sum underflow. Required inner products still use ordinary float64 arithmetic: an unusable product returns `non_finite`, even if the inputs and reported residual are finite. This is deliberate failure reporting, not arbitrary-scale arithmetic support. SPD remains a caller precondition.

## Validation and future backends

GoogleTest covers native ownership, aliasing, kernels, CSR validation, and CG/PCG breakdown. Pytest covers conversion, strict arrays, snapshots, GIL/workspace behavior, and the benchmark contract. CMake presets isolate serial, OpenMP, debug, and sanitizer builds. CI checks formatting, static analysis, native/Python tests, sparse comparison artifacts, and installed-wheel imports. Timing thresholds require a stable dedicated machine.

CUDA is a later extension. It needs explicit device storage/views, stream and completion semantics, device workspace, and separate transfer/setup timing. CPU borrowed pointers must not be reinterpreted as device pointers. The operation boundary and shared recurrence provide the extension points; add the backend abstraction when a second implementation exists, with explicit device lifetime and synchronization contracts.
