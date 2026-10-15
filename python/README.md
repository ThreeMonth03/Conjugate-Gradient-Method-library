# Solver interfaces

Use CPython 3.12.x and import `cgpy` after installing the project into the local `.venv` as described in the [main README](../README.md). The CPU interface solves `Ax = b` for real symmetric positive-definite matrices with float64 arithmetic.

## Convenience solve

```python
result = cgpy.solve(
    A, b, x0=None,
    rtol=1e-8, atol=0.0, maxiter=1000,
    num_threads=1, algorithm="standard", fused=True,
    residual_refresh=50, preconditioner="none", dense_kernel="native",
)
```

`A` is a nonempty square dense array or `cgpy.CsrMatrix`. `b` and `x0` are one-dimensional vectors matching its size. Dense array-like inputs are normalized to aligned C-contiguous float64 arrays. Compatible inputs can be borrowed for this synchronous call, so do not modify them concurrently. The output owns its storage; `A`, `b`, and `x0` are preserved. An omitted `x0` selects a zero initial guess.

| Option | Contract |
| --- | --- |
| `rtol`, `atol` | Finite, nonnegative tolerances; stopping threshold is `max(atol, rtol * ||b||2)`. |
| `maxiter` | Maximum iterations; zero only checks the initial guess. |
| `num_threads` | Positive integer; values greater than one require an OpenMP build. |
| `algorithm` | `"standard"` uses the residual recurrence; `"explicit"` recomputes the residual each iteration as an explicit-residual reference formulation. |
| `fused` | Select fused CPU operations or separate operations for comparison. This does not select the CG recurrence. |
| `residual_refresh` | Standard CG periodically recomputes the true residual and restarts the direction. Zero disables periodic refresh; convergence and final residual checks remain enabled. |
| `preconditioner` | `"none"` selects CG; `"jacobi"` selects PCG with the inverse matrix diagonal and requires `algorithm="standard"`. |
| `dense_kernel` | Explicit `"native"` or `"blas"` dense matrix-vector kernel. BLAS requires an enabled build, dense storage and `num_threads=1`; CSR requires `"native"`. |

Jacobi setup rejects missing, zero, negative, or non-finite diagonal entries and non-finite inverse diagonals. It does not prove that the full matrix is positive definite.

`result.x` is the solution. `result.info` reports `status`, `converged`, `iterations`, `matvecs`, `residual_norm`, and `restarts`. The result also exposes convenience properties such as `result.converged` and `result.residual_norm`. Matvec counts include initial, replacement, and final verification operations when performed.

| Status | Meaning |
| --- | --- |
| `converged` | The independently evaluated true residual satisfies the requested threshold. |
| `max_iterations` | The iteration budget was exhausted. |
| `non_positive_curvature` | A search direction produced negative curvature; the SPD precondition may not hold. |
| `non_finite` | A value became non-finite or a required inner product was outside the usable double range, including zero or subnormal curvature. The reported true residual can still be finite. |

Invalid dimensions, layouts, options, or preconditioner setup raise an exception. Numerical breakdown and iteration limits return a status; inspect it even when the output contains finite numbers.

## Repeated and in-place solves

```python
import numpy as np
import cgpy

A = np.array([[4.0, 1.0], [1.0, 3.0]])
b = np.array([1.0, 2.0])
prepared = cgpy.prepare(A, num_threads=1, preconditioner="jacobi")
workspace = prepared.create_workspace()
x = np.zeros_like(b)
info = prepared.solve_into(b, x, workspace=workspace, rtol=1e-10)
assert info.converged

x.fill(0.0)  # Restore the initial guess before the next solve.
info = prepared.solve_into(b, x, workspace=workspace, rtol=1e-10)
```

`prepare` owns a snapshot of the matrix and caches the selected preconditioner. Changes to the original matrix do not change the prepared solver. Its thread count, preconditioner and `dense_kernel` are selected at preparation time; each call independently selects `fused`. The workspace owns the solver's reusable vectors and must match the system size.

For an existing dense array or CSR operator, the strict function is:

```python
workspace = cgpy.Workspace(b.size)
info = cgpy.solve_into(
    A, b, x, workspace=workspace,
    num_threads=1, rtol=1e-8, atol=0.0, maxiter=1000,
    algorithm="standard", fused=True, residual_refresh=50,
    preconditioner="none", dense_kernel="native",
)
```

This function treats `x` as both initial guess and output and returns `SolveInfo`. Dense `A`, `b`, and `x` must already be aligned C-contiguous float64 NumPy arrays; `x` must be writable. It performs no implicit dtype or layout conversion. Keep `x` disjoint from the matrix and `b`. Inputs and output must remain alive and free from conflicting mutation until the synchronous call returns. A workspace cannot serve overlapping calls; concurrent solves require separate workspaces and output arrays. Long native computation releases the Python GIL.

Use `prepare` to reuse matrix and Jacobi setup. A standalone `solve` or `solve_into` call with `preconditioner="jacobi"` prepares the diagonal for that call.

BLAS delegates dense matrix-vector multiplication and retains the chosen CPU vector update/reduction policy. It does not select the algorithm or preconditioner. Check `cgpy.blas_enabled()` before selecting it; `cgpy.build_info()['blas']` records its provider, configuration, integer width, sequential thread mode and private symbol prefix. See the [pip/build instructions](../docs/install.rst) and generated [Sphinx API](../docs/api.rst) for the complete docstrings and examples.

## Sparse matrices

```python
A = cgpy.CsrMatrix(
    indptr=np.array([0, 2, 4], dtype=np.int64),
    indices=np.array([0, 1, 0, 1], dtype=np.int64),
    data=np.array([4.0, 1.0, 1.0, 3.0], dtype=np.float64),
    shape=(2, 2),
)
result = cgpy.solve(A, b, preconditioner="jacobi", rtol=1e-10)
assert result.converged
```

CSR stores row pointers, column indices, and values without converting the operator to dense storage. Row pointers start at zero, are monotone, and end at the number of values. Column indices must be in range, sorted within each row, and unique. Canonicalize a SciPy CSR matrix with `sum_duplicates()` and `sort_indices()` before wrapping its arrays if needed.

## CUDA interface

`from cgpy import cuda` works with CPU-only wheels. `cuda.enabled()` checks whether the optional extension/runtime loads; `cuda.is_available(device=0)` checks a usable GPU. Both CPU and CUDA interfaces require explicit backend selection. See the [CUDA guide](../docs/cuda.rst) for a project-local toolkit, separate `.venv-cuda`, source/wheel installs and architecture settings.

```python
from cgpy import cuda

device_b = cuda.Vector.from_host(b, device=0)
prepared = cuda.prepare_host(
    np.array([[4.0, 1.0], [1.0, 3.0]]),
    device=0, dense_kernel="native", preconditioner="jacobi",
)
workspace = prepared.create_workspace()
device_x = prepared.create_vector()
info = prepared.solve_into(device_b, device_x, workspace=workspace, rtol=1e-10)
assert info.converged
print(device_x.to_host())
```

Host transfer is explicit. `cuda.prepare` accepts a device matrix; `prepare_host` uploads and snapshots host dense or `cgpy.CsrMatrix` storage. Device `solve` returns `cuda.SolveResult` with an owned `Vector`. Device `solve_into` borrows compatible CUDA Array Interface arrays and uses supplied output/workspace. `dense_kernel="native"` uses project kernels; `"cublas"` selects dense GEMV and rejects CSR. Jacobi supports `algorithm="standard"`; vector fusion remains an independent option.

Calls finish on the selected borrowed stream before returning. Keep nondefault streams alive, and use a distinct workspace/output for each overlapping solve. External arrays must be contiguous float64, writable for output, disjoint and free from conflicting mutation. CUDA Array Interface v3 producer dependencies are honored; v2 producers need caller-established readiness or explicit synchronization. No call changes PyTorch's current stream or converts host pointers to device pointers.

CUDA benchmark prepared/API scopes use resident inputs and exclude host upload and verification download. They measure completed wall time, including host dispatch and iteration scalar synchronization. Transfer/setup metadata is reported separately; neither scope measures end-to-end host-to-host performance.

## C++ core

The native source interface uses feature headers under `cpp/cg/` and the CMake `cg_core` target. Views borrow caller-owned storage; keep it alive until the synchronous solve returns. The workspace and output must be disjoint from the inputs. The default native algorithm is the standard recurrence; the explicit-residual reference algorithm is opt-in.

```cpp
#include <cg/cg.hpp>
#include <utility>
#include <vector>

int main()
{
    cg::DenseMatrix A(2, 2, {4.0, 1.0, 1.0, 3.0});
    std::vector<double> b{1.0, 2.0};
    std::vector<double> x(2);
    cg::PreparedSolver plan(std::move(A), 1, cg::PreconditionerKind::jacobi);
    auto workspace = plan.create_workspace();
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    auto info = plan.solve_into(b, x, workspace, options);
    return info.status == cg::Status::converged ? 0 : 1;
}
```

Select `cg::PreconditionerKind::none` for plain CG. `cg::CsrMatrix` accepts owning row-pointer, column-index, and value vectors and constructs the same `cg::PreparedSolver` type. Prepared ownership and Jacobi setup live in the C++ core. `cg::solve_into` also accepts borrowed operators for one-shot solves. `CpuBackend` contains execution options; matrix and vector storage do not contain process-wide thread settings. Workspace reuse avoids repeated allocation of solver vectors; the backend and runtime can still perform their own initialization.

## Feature modules

`cgpy` provides the public convenience exports. Implementation follows domain
boundaries: `cgpy.matrix` owns matrix conversion, `cgpy.storage` groups device storage
and transfers, and `cgpy.solver` owns solve functions, workspace and result
types.
`cgpy._runtime` loads native modules, `_interop` handles host representation,
and `solver/_validation.py` checks solve policy. `cgpy.cuda` is the
explicit device facade. The private extensions are `cgpy._core` and
`cgpy._cuda`. Internal modules import their peers rather than reaching through
the package root.

The supported interface is modern linear CG and Jacobi PCG. The earlier
`_cgpy`, `utils`, matrix subclasses and nonlinear methods have been removed.
