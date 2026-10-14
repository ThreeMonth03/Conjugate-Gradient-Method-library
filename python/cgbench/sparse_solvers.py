"""Sparse direct-solver candidates kept separate from CG implementation ratios."""

import time

from .backends import Outcome
from .numerics import l2_from_square


class ScipyKrylovExecutor:
    """Official SciPy CG/PCG with its own recurrence and allocation policy.

    Instrument iteration/matvec counts only on the first untimed correctness
    call. Formal timing uses public CSR input without callback or wrappers.
    """

    def __init__(self, arrays, options, scope, jacobi):
        import numpy as np
        from scipy.sparse.linalg import LinearOperator, cg

        self.np = np
        self.cg = cg
        self.LinearOperator = LinearOperator
        matrix, self.rhs, self.initial = arrays
        self.matrix = matrix.copy() if scope == "prepared" else matrix
        self.options, self.scope, self.jacobi = options, scope, jacobi
        self.x = np.empty_like(self.initial) if scope == "prepared" else None
        self.iterations = self.matvecs = 0
        self.profile_first_call = True
        self.preconditioner = None
        self.setup_ns = 0
        if jacobi:
            start = time.perf_counter_ns()
            probe = self._make_preconditioner()
            self.setup_ns = time.perf_counter_ns() - start
            if scope == "prepared":
                self.preconditioner = probe

    def _make_preconditioner(self):
        np = self.np
        inverse = self.matrix.diagonal()
        if not np.all(np.isfinite(inverse)) or not np.all(inverse > 0):
            raise ValueError("Jacobi requires finite positive diagonal entries")
        np.reciprocal(inverse, out=inverse)
        if not np.all(np.isfinite(inverse)) or not np.all(inverse > 0):
            raise ValueError("Jacobi reciprocal must be finite and positive")
        return self.LinearOperator(
            self.matrix.shape, matvec=lambda vector: inverse * vector, dtype=np.float64
        )

    def __call__(self):
        np = self.np
        x = self.x if self.x is not None else np.empty_like(self.initial)
        np.copyto(x, self.initial)
        preconditioner = self.preconditioner
        if self.jacobi and self.scope == "api":
            preconditioner = self._make_preconditioner()
        callback = None
        operator = self.matrix
        if self.profile_first_call:
            self.iterations = self.matvecs = 0

            def callback(_x):
                self.iterations += 1

            def matvec(vector):
                self.matvecs += 1
                return self.matrix @ vector

            operator = self.LinearOperator(self.matrix.shape, matvec=matvec, dtype=np.float64)
        result, info = self.cg(
            operator,
            self.rhs,
            x0=x,
            rtol=self.options["rtol"],
            atol=self.options["atol"],
            maxiter=self.options["maxiter"],
            M=preconditioner,
            callback=callback,
        )
        if self.profile_first_call:
            self.matvecs += 1  # Independent final true-residual evaluation below.
        self.profile_first_call = False
        np.copyto(x, result)
        residual = self.rhs - self.matrix @ x
        norm = l2_from_square(float(np.dot(residual, residual)), residual)
        status = "converged" if info == 0 else "max_iterations"
        if not np.isfinite(norm):
            status = "non_finite"
        return Outcome(x, status, self.iterations, self.matvecs, norm)

    def benchmark_metadata(self):
        return {
            "implementation_role": "algorithm_comparison",
            "solver_family": "pcg" if self.jacobi else "cg",
            "library_solver": "scipy.sparse.linalg.cg",
            "residual_policy": "SciPy library recurrence; no configured periodic replacement; independent final true-residual gate",
            "iteration_counts_measurement": "first untimed correctness call; timed calls use CSR directly without callback",
            "preconditioner": "jacobi" if self.jacobi else "none",
            "preconditioner_setup_elapsed_ns": self.setup_ns,
            "preconditioner_stored_bytes": int(self.initial.nbytes) if self.jacobi else 0,
            "setup_in_timed_call": self.scope == "api",
            "setup_measurement": "single untimed setup probe",
            "workspace_policy": "SciPy allocates its own iterative workspace in both scopes",
        }


class SparseLUExecutor:
    """Compare factor+solve and solve with an already computed SuperLU factor.

    CSR-to-CSC conversion is excluded and reported separately for both. The
    first correctness call profiles setup phases; formal timed calls do not
    contain additional phase timers or factor-storage introspection.
    """

    def __init__(self, arrays, options, scope, reuse):
        import numpy as np
        from scipy.sparse.linalg import splu

        self.np = np
        self.splu = splu
        self.matrix, self.rhs, self.initial = arrays
        if not hasattr(self.matrix, "indptr"):
            raise ValueError("sparse LU requires CSR inputs")
        self.options = options
        self.scope = scope
        self.reuse = reuse
        self.x = np.empty_like(self.initial) if scope == "prepared" else None
        start = time.perf_counter_ns()
        self.csc = self.matrix.tocsc()
        self.conversion_ns = time.perf_counter_ns() - start
        self.factorization_ns = None
        self.factor = None
        self.last_factor = None
        self.factor_bytes = None
        self.profile_first_call = True
        if reuse:
            start = time.perf_counter_ns()
            self.factor = splu(self.csc)
            self.factorization_ns = time.perf_counter_ns() - start

    def __call__(self):
        np = self.np
        x = self.x if self.x is not None else np.empty_like(self.initial)
        # Direct LU does not use an initial guess mathematically; retain the
        # same charged output reset as iterative candidates for comparability.
        np.copyto(x, self.initial)
        if self.reuse:
            factor = self.factor
        elif self.profile_first_call:
            start = time.perf_counter_ns()
            factor = self.splu(self.csc)
            self.factorization_ns = time.perf_counter_ns() - start
        else:
            factor = self.splu(self.csc)
        if self.profile_first_call:
            self.last_factor = factor
        self.profile_first_call = False
        np.copyto(x, factor.solve(self.rhs))
        residual = self.rhs - self.matrix @ x
        true_norm = l2_from_square(float(np.dot(residual, residual)), residual)
        b_norm = l2_from_square(float(np.dot(self.rhs, self.rhs)), self.rhs)
        threshold = max(self.options["atol"], self.options["rtol"] * b_norm)
        status = (
            "converged"
            if np.isfinite(true_norm) and true_norm <= threshold
            else "non_finite"
            if not np.isfinite(true_norm)
            else "max_iterations"
        )
        return Outcome(x, status, 0, 1, true_norm)

    @staticmethod
    def _sparse_bytes(matrix):
        return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)

    def benchmark_metadata(self):
        factor = self.last_factor if self.last_factor is not None else self.factor
        if self.factor_bytes is None and factor is not None:
            # L/U conversion is intentionally outside all timed blocks.
            self.factor_bytes = self._sparse_bytes(factor.L) + self._sparse_bytes(factor.U)
            self.factor_bytes += factor.perm_r.nbytes + factor.perm_c.nbytes
        if not self.reuse:
            # Keep only measured storage counts, not a previous factor while
            # timing construction of a new one on the next invocation.
            self.last_factor = None
        return {
            "implementation_role": "algorithm_comparison",
            "solver_family": "sparse_lu_reuse" if self.reuse else "sparse_lu_total",
            "preconditioner": "none",
            "preconditioner_setup_elapsed_ns": 0,
            "preconditioner_stored_bytes": 0,
            "format_conversion_elapsed_ns": self.conversion_ns,
            "csc_stored_bytes": self._sparse_bytes(self.csc),
            "factorization_setup_elapsed_ns": self.factorization_ns,
            "factor_stored_bytes": int(self.factor_bytes)
            if self.factor_bytes is not None
            else None,
            "factorization_in_timed_call": not self.reuse,
            "setup_measurement": "single untimed setup/correctness probe",
        }
