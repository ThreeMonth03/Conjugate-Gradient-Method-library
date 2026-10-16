"""Independent CG/PCG policy using a structural array-operations contract."""

import math
import sys
import time

from .. import execution, numerics
from .operations import ArrayOperations


class ReferenceExecutor:
    """Use the native solver's recurrence, restart, and true-residual policy."""

    def __init__(self, ops: ArrayOperations, options, scope):
        self.ops = ops
        self.options = options
        self.scope = scope
        self.preconditioner = options.get("preconditioner", "none")
        if self.preconditioner not in ("none", "jacobi"):
            raise ValueError("preconditioner must be none or jacobi")
        if (
            self.preconditioner == "jacobi"
            and options["algorithm"] != "standard"
        ):
            raise ValueError("Jacobi PCG requires the standard recurrence")
        # Prepare a snapshot once, matching the native prepared operator.
        if scope == "prepared":
            self.ops.snapshot_matrix()
        self.storage = ops.workspace() if scope == "prepared" else None
        if scope == "session":
            # Cache only iteration vectors, never the returned solution.
            self.storage = [ops.empty_vector() for _ in range(4)]
            if self.preconditioner == "jacobi":
                self.storage.append(ops.empty_vector())
        self.inverse = None
        self.preconditioner_setup_ns = 0
        if self.preconditioner == "jacobi":
            start = time.perf_counter_ns()
            probe = ops.inverse_diagonal()
            self.preconditioner_setup_ns = time.perf_counter_ns() - start
            if scope == "prepared":
                self.inverse = probe
                self.storage.append(ops.empty_vector())

    def benchmark_metadata(self):
        return {
            "implementation_role": "same_policy_reference",
            "solver_family": "pcg"
            if self.preconditioner == "jacobi"
            else "cg",
            "preconditioner": self.preconditioner,
            "preconditioner_setup_elapsed_ns": self.preconditioner_setup_ns,
            "preconditioner_stored_bytes": self.ops.initial_nbytes
            if self.preconditioner == "jacobi"
            else 0,
            "setup_in_timed_call": self.scope in ("api", "session"),
            "setup_measurement": "single untimed setup probe",
            "residual_policy": self.options["algorithm"]
            + " with residual_refresh="
            + str(self.options["residual_refresh"]),
        }

    def __call__(self):
        with self.ops.execution_context():
            return self._solve()

    def _solve(self):
        ops = self.ops
        storage = (
            [ops.empty_vector(), *self.storage]
            if self.scope == "session"
            else self.storage
            if self.storage is not None
            else ops.workspace()
        )
        inverse = self.inverse
        if self.preconditioner == "jacobi" and self.scope in (
            "api",
            "session",
        ):
            inverse = ops.inverse_diagonal()
            if self.scope == "api":
                storage.append(ops.empty_vector())
        x, residual, direction, product, scratch = storage[:5]
        z = storage[5] if inverse is not None else None
        ops.copy(ops.initial, x)
        explicit = self.options["algorithm"] == "explicit"

        def norm(squared, vector):
            # Do not export Torch tensors on the ordinary squared-norm path.
            if squared >= sys.float_info.min and math.isfinite(squared):
                return math.sqrt(squared)
            return numerics.l2_from_square(squared, ops.export(vector))

        b_norm = norm(ops.dot(ops.rhs, ops.rhs), ops.rhs)
        tolerance = max(self.options["atol"], self.options["rtol"] * b_norm)
        matvecs = 0
        iterations = 0
        restarts = 0

        def true_residual():
            nonlocal matvecs
            # Preserve Ap for the explicit algorithm's direction update.
            ops.residual(x, residual, scratch, explicit)
            matvecs += 1
            return ops.dot(residual, residual)

        def result(status, rho):
            residual_norm = norm(rho, residual)
            # CUDA keeps the solution resident until an untimed correctness
            # gate explicitly downloads it. CPU exports retain their behavior.
            output = ops.outcome_x(x)
            return execution.Outcome(
                output, status, iterations, matvecs, residual_norm, restarts
            )

        def finish(status):
            final_rho = rho if explicit else true_residual()
            final_norm = norm(final_rho, residual)
            if not math.isfinite(final_norm):
                status = "non_finite"
            elif status != "non_finite" and final_norm <= tolerance:
                status = "converged"
            return result(status, final_rho)

        rho = true_residual()
        residual_norm = norm(rho, residual)
        if not math.isfinite(residual_norm) or not math.isfinite(b_norm):
            return result("non_finite", rho)
        if residual_norm <= tolerance:
            return result("converged", rho)
        if self.options["maxiter"] == 0:
            return result("max_iterations", rho)
        if not math.isfinite(rho) or rho < sys.float_info.min:
            return result("non_finite", rho)
        if inverse is None:
            ops.initialize_direction(residual, direction, explicit)
        else:
            ops.precondition(inverse, residual, z)
            rho = ops.dot(residual, z)
            ops.initialize_direction(z, direction, False)
            if not math.isfinite(rho) or rho < sys.float_info.min:
                return finish("non_finite")
        for iteration in range(self.options["maxiter"]):
            ops.matvec(direction, product)
            matvecs += 1
            denominator = ops.dot(direction, product)
            if not math.isfinite(denominator):
                return finish("non_finite")
            if denominator < 0.0:
                return finish("non_positive_curvature")
            if denominator < sys.float_info.min:
                return finish("non_finite")
            alpha = (
                -ops.dot(residual, direction) / denominator
                if explicit
                else rho / denominator
            )
            if not math.isfinite(alpha):
                return finish("non_finite")
            ops.axpy(alpha, direction, x, scratch)
            iterations = iteration + 1
            if explicit:
                rho_next = true_residual()
                norm_next = norm(rho_next, residual)
                if not math.isfinite(norm_next):
                    return result("non_finite", rho_next)
                if norm_next <= tolerance:
                    return result("converged", rho_next)
                if (
                    not math.isfinite(rho_next)
                    or rho_next < sys.float_info.min
                ):
                    return result("non_finite", rho_next)
                if iterations == self.options["maxiter"]:
                    return result("max_iterations", rho_next)
                rho = rho_next
                beta = ops.dot(residual, product) / denominator
            else:
                ops.axpy(-alpha, product, residual, scratch)
                rho_next = ops.dot(residual, residual)
                norm_next = norm(rho_next, residual)
                if not math.isfinite(norm_next):
                    return finish("non_finite")
                candidate = norm_next <= tolerance
                # Match the native engine: an unusable noncandidate squared
                # residual fails before a scheduled checkpoint can recover it.
                if not candidate and (
                    not math.isfinite(rho_next)
                    or rho_next < sys.float_info.min
                ):
                    return finish("non_finite")
                refresh = self.options["residual_refresh"]
                if (
                    candidate
                    or (refresh and iterations % refresh == 0)
                    or iterations == self.options["maxiter"]
                ):
                    rho_next = true_residual()
                    norm_next = norm(rho_next, residual)
                    if not math.isfinite(norm_next):
                        return result("non_finite", rho_next)
                    if norm_next <= tolerance:
                        return result("converged", rho_next)
                    if (
                        not math.isfinite(rho_next)
                        or rho_next < sys.float_info.min
                    ):
                        return result("non_finite", rho_next)
                    if iterations == self.options["maxiter"]:
                        return result("max_iterations", rho_next)
                    # True residual replacement restarts the search direction.
                    if inverse is None:
                        ops.initialize_direction(residual, direction, False)
                        rho = rho_next
                    else:
                        ops.precondition(inverse, residual, z)
                        rho = ops.dot(residual, z)
                        if not math.isfinite(rho) or rho < sys.float_info.min:
                            return result("non_finite", rho_next)
                        ops.initialize_direction(z, direction, False)
                    restarts += 1
                    continue
                if inverse is not None:
                    ops.precondition(inverse, residual, z)
                    rho_next = ops.dot(residual, z)
                    if (
                        not math.isfinite(rho_next)
                        or rho_next < sys.float_info.min
                    ):
                        return finish("non_finite")
                beta = rho_next / rho
            if not math.isfinite(beta):
                return finish("non_finite")
            ops.direction(
                beta, residual if inverse is None else z, direction, explicit
            )
            rho = rho_next
        rho = true_residual()
        if not math.isfinite(rho):
            return result("non_finite", rho)
        return result(
            "converged" if math.sqrt(rho) <= tolerance else "max_iterations",
            rho,
        )
