# Solver interfaces

Import `cgpy` after installing the project into the local `.venv` as described in the [main README](../README.md). The CPU interface solves `Ax = b` for real symmetric positive-definite matrices with float64 arithmetic.

## Convenience solve

```python
result = cgpy.solve(
    A, b, x0=None,
    rtol=1e-8, atol=0.0, maxiter=1000,
    num_threads=1, algorithm="standard", fused=True,
    residual_refresh=50, preconditioner="none",
)
```

`A` is a nonempty square dense array or `cgpy.CSRMatrix`. `b` and `x0` are one-dimensional vectors matching its size. Dense array-like inputs are normalized to C-contiguous float64 arrays. Compatible inputs can be borrowed for this synchronous call, so do not modify them concurrently. The output owns its storage; `A`, `b`, and `x0` are preserved. An omitted `x0` selects a zero initial guess.

| Option | Contract |
| --- | --- |
| `rtol`, `atol` | Finite, nonnegative tolerances; stopping threshold is `max(atol, rtol * ||b||2)`. |
| `maxiter` | Maximum iterations; zero only checks the initial guess. |
| `num_threads` | Positive integer; values greater than one require an OpenMP build. |
| `algorithm` | `"standard"` uses the residual recurrence; `"explicit"` recomputes the residual each iteration using the legacy formulation. |
| `fused` | Select fused CPU operations or separate operations for comparison. This does not select the CG recurrence. |
| `residual_refresh` | Standard CG periodically recomputes the true residual and restarts the direction. Zero disables periodic refresh; convergence and final residual checks remain enabled. |
| `preconditioner` | `"none"` selects CG; `"jacobi"` selects PCG with the inverse matrix diagonal and requires `algorithm="standard"`. |

Jacobi setup rejects missing, zero, negative, or non-finite diagonal entries and non-finite inverse diagonals. It does not prove that the full matrix is positive definite.

`result.x` is the solution. `result.info` reports `status`, `converged`, `iterations`, `matvecs`, `residual_norm`, and `restarts`. The result also exposes convenience properties such as `result.converged` and `result.residual_norm`. Matvec counts include initial, replacement, and final verification operations when performed.

| Status | Meaning |
| --- | --- |
| `converged` | The independently evaluated true residual satisfies the requested threshold. |
| `max_iterations` | The iteration budget was exhausted. |
| `non_positive_curvature` | A search direction produced non-positive curvature; the SPD precondition may not hold. |
| `non_finite` | The computation encountered a non-finite value. |

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

`prepare` owns a snapshot of the matrix and caches the selected preconditioner. Changes to the original matrix do not change the prepared solver. Its thread count and preconditioner are selected at preparation time. The workspace owns the solver's reusable vectors and must match the system size.

For an existing dense array or CSR operator, the strict function is:

```python
workspace = cgpy.Workspace(b.size)
info = cgpy.solve_into(
    A, b, x, workspace=workspace,
    num_threads=1, rtol=1e-8, atol=0.0, maxiter=1000,
    algorithm="standard", fused=True, residual_refresh=50,
    preconditioner="none",
)
```

This function treats `x` as both initial guess and output and returns `SolveInfo`. Dense `A`, `b`, and `x` must already be C-contiguous float64 NumPy arrays; `x` must be writable. It performs no implicit dtype or layout conversion. Keep `x` disjoint from the matrix and `b`. Inputs and output must remain alive and free from conflicting mutation until the synchronous call returns. A workspace cannot serve overlapping calls; concurrent solves require separate workspaces and output arrays. Long native computation releases the Python GIL.

Use `prepare` to reuse matrix and Jacobi setup. A standalone `solve` or `solve_into` call with `preconditioner="jacobi"` prepares the diagonal for that call.

## Sparse matrices

```python
A = cgpy.CSRMatrix(
    indptr=np.array([0, 2, 4], dtype=np.int64),
    indices=np.array([0, 1, 0, 1], dtype=np.int64),
    data=np.array([4.0, 1.0, 1.0, 3.0], dtype=np.float64),
    shape=(2, 2),
)
result = cgpy.solve(A, b, preconditioner="jacobi", rtol=1e-10)
assert result.converged
```

CSR stores row pointers, column indices, and values without converting the operator to dense storage. Row pointers start at zero, are monotone, and end at the number of values. Column indices must be in range, sorted within each row, and unique. Canonicalize a SciPy CSR matrix with `sum_duplicates()` and `sort_indices()` before wrapping its arrays if needed.

## C++ core

The native source interface uses the headers under `include/cg/` and the CMake `cg_core` target. Views borrow caller-owned storage; keep it alive until the synchronous solve returns. The workspace and output must be disjoint from the inputs. The example explicitly selects the recurrence because the native options retain the legacy explicit algorithm as their default.

```cpp
#include <cg/cg.hpp>
#include <vector>

int main()
{
    cg::DenseMatrix A(2, 2, {4.0, 1.0, 1.0, 3.0});
    cg::Vector b(std::vector<double>{1.0, 2.0});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    cg::CpuBackend backend(1, true);
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    cg::JacobiPreconditioner preconditioner(A.view());
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace,
                               backend, options, &preconditioner);
    return info.status == cg::Status::converged ? 0 : 1;
}
```

Omit the final preconditioner pointer for plain CG. `cg::CsrMatrix` accepts owning row-pointer, column-index, and value vectors and works with the same native solver interface. `CpuBackend` contains execution options; matrix and vector storage do not contain process-wide thread settings. Core operations include matvec, dot, vector updates, and selected fused operations. Workspace reuse avoids repeated allocation of solver vectors; the backend and runtime can still perform their own initialization.

## Legacy methods

The installed `utils` module and `_cgpy.Matrix` / `_cgpy.CG` bindings retain the earlier APIs, including nonlinear Fletcher-Reeves, Dai-Yuan, and Hager-Zhang methods. For example:

```python
import utils

x = utils.custom_linear_CG(
    x=np.zeros_like(b), a=np.array([[4.0, 1.0], [1.0, 3.0]]),
    b=b, epsilon=1e-10, epoch=1000, num_threads=1,
)
```

Legacy `epsilon` is an absolute residual tolerance. Use the new API when comparing explicit and recurrence algorithms, fusion, workspace reuse, or Jacobi PCG under a shared stopping rule.
