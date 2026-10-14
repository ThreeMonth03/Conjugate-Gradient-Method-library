"""Preallocated NumPy/PyTorch references and the public native API adapter."""

import math
from dataclasses import dataclass


@dataclass
class Outcome:
    x: object
    status: str
    iterations: int
    matvecs: int
    residual_norm: float

    @property
    def converged(self):
        return self.status == "converged"


class NativeExecutor:
    def __init__(self, arrays, options, scope, threads):
        import cgpy
        import numpy as np

        self.np = np
        self.cgpy = cgpy
        self.matrix, self.rhs, self.initial = arrays
        self.options = options
        self.scope = scope
        self.threads = threads
        if scope == "prepared":
            self.solver = cgpy.prepare(self.matrix, num_threads=threads)
            self.workspace = self.solver.create_workspace()
            self.x = np.empty_like(self.initial)

    def __call__(self):
        if self.scope == "prepared":
            # Reset is part of EVERY measured invocation, including repetitions.
            self.np.copyto(self.x, self.initial)
            info = self.solver.solve_into(
                self.rhs, self.x, workspace=self.workspace, **self.options
            )
            x = self.x
        else:
            info = self.cgpy.solve(
                self.matrix,
                self.rhs,
                x0=self.initial,
                num_threads=self.threads,
                **self.options,
            )
            x = info.x
        return Outcome(
            x,
            str(info.status),
            info.iterations,
            info.matvecs,
            info.residual_norm,
        )


class NumpyOps:
    def __init__(self, arrays):
        import numpy as np

        self.np = np
        self.matrix, self.rhs, self.initial = arrays

    def workspace(self):
        return [self.np.empty_like(self.initial) for _ in range(5)]

    def validate(self, api):
        arrays = (self.matrix, self.rhs, self.initial) if api else (self.rhs, self.initial)
        if not all(self.np.all(self.np.isfinite(array)) for array in arrays):
            raise ValueError("inputs must be finite")

    def copy(self, source, destination):
        self.np.copyto(destination, source)

    def matvec(self, vector, out):
        self.np.matmul(self.matrix, vector, out=out)

    def dot(self, left, right):
        return float(self.np.dot(left, right))

    def residual(self, x, residual, product, explicit):
        self.matvec(x, product)
        if explicit:
            self.np.subtract(product, self.rhs, out=residual)
        else:
            self.np.subtract(self.rhs, product, out=residual)

    def axpy(self, alpha, source, destination, scratch):
        self.np.multiply(source, alpha, out=scratch)
        self.np.add(destination, scratch, out=destination)

    def direction(self, beta, residual, direction, explicit):
        self.np.multiply(direction, beta, out=direction)
        if explicit:
            self.np.subtract(direction, residual, out=direction)
        else:
            self.np.add(direction, residual, out=direction)

    def initialize_direction(self, residual, direction, explicit):
        self.copy(residual, direction)
        if explicit:
            self.np.negative(direction, out=direction)

    def export(self, x):
        return x


class TorchOps:
    def __init__(self, arrays, threads):
        import torch

        torch.set_num_threads(threads)
        # The worker calls this only once, before any interop work starts.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        self.torch = torch
        self.matrix, self.rhs, self.initial = [torch.from_numpy(array) for array in arrays]

    def workspace(self):
        return [self.torch.empty_like(self.initial) for _ in range(5)]

    def validate(self, api):
        arrays = (self.matrix, self.rhs, self.initial) if api else (self.rhs, self.initial)
        if not all(bool(self.torch.isfinite(array).all()) for array in arrays):
            raise ValueError("inputs must be finite")

    @staticmethod
    def copy(source, destination):
        destination.copy_(source)

    def matvec(self, vector, out):
        self.torch.mv(self.matrix, vector, out=out)

    def dot(self, left, right):
        return float(self.torch.dot(left, right))

    def residual(self, x, residual, product, explicit):
        self.matvec(x, product)
        if explicit:
            self.torch.sub(product, self.rhs, out=residual)
        else:
            self.torch.sub(self.rhs, product, out=residual)

    @staticmethod
    def axpy(alpha, source, destination, scratch):
        destination.add_(source, alpha=alpha)

    @staticmethod
    def direction(beta, residual, direction, explicit):
        direction.mul_(beta)
        direction.add_(residual, alpha=-1.0 if explicit else 1.0)

    @staticmethod
    def initialize_direction(residual, direction, explicit):
        direction.copy_(residual)
        if explicit:
            direction.neg_()

    @staticmethod
    def export(x):
        return x.numpy()


class ReferenceExecutor:
    """Use the native solver's recurrence, restart, and true-residual policy."""

    def __init__(self, ops, options, scope):
        self.ops = ops
        self.options = options
        self.scope = scope
        # Prepare a snapshot once, matching the native prepared operator.
        if scope == "prepared":
            self.ops.matrix = (
                self.ops.matrix.copy() if isinstance(ops, NumpyOps) else self.ops.matrix.clone()
            )
        self.storage = ops.workspace() if scope == "prepared" else None

    def __call__(self):
        ops = self.ops
        if isinstance(ops, TorchOps):
            with ops.torch.inference_mode():
                return self._solve()
        return self._solve()

    def _solve(self):
        ops = self.ops
        ops.validate(self.scope == "api")
        x, residual, direction, product, scratch = (
            self.storage if self.storage is not None else ops.workspace()
        )
        ops.copy(ops.initial, x)
        explicit = self.options["algorithm"] == "explicit"
        b_norm = math.sqrt(ops.dot(ops.rhs, ops.rhs))
        tolerance = max(self.options["atol"], self.options["rtol"] * b_norm)
        matvecs = 0
        iterations = 0

        def true_residual():
            nonlocal matvecs
            # Preserve Ap for the explicit algorithm's direction update.
            ops.residual(x, residual, scratch, explicit)
            matvecs += 1
            return ops.dot(residual, residual)

        def result(status, rho):
            norm = math.sqrt(rho) if rho >= 0.0 else float("nan")
            return Outcome(ops.export(x), status, iterations, matvecs, norm)

        def finish(status):
            final_rho = rho if explicit else true_residual()
            if not math.isfinite(final_rho):
                status = "non_finite"
            elif status != "non_finite" and math.sqrt(final_rho) <= tolerance:
                status = "converged"
            return result(status, final_rho)

        rho = true_residual()
        if not math.isfinite(rho) or not math.isfinite(b_norm):
            return result("non_finite", rho)
        if math.sqrt(rho) <= tolerance:
            return result("converged", rho)
        if self.options["maxiter"] == 0:
            return result("max_iterations", rho)
        ops.initialize_direction(residual, direction, explicit)
        for iteration in range(self.options["maxiter"]):
            ops.matvec(direction, product)
            matvecs += 1
            denominator = ops.dot(direction, product)
            if not math.isfinite(denominator):
                return finish("non_finite")
            if denominator <= 0.0:
                return finish("non_positive_curvature")
            alpha = -ops.dot(residual, direction) / denominator if explicit else rho / denominator
            if not math.isfinite(alpha):
                return finish("non_finite")
            ops.axpy(alpha, direction, x, scratch)
            iterations = iteration + 1
            if explicit:
                rho_next = true_residual()
                if not math.isfinite(rho_next):
                    return result("non_finite", rho_next)
                if math.sqrt(rho_next) <= tolerance:
                    return result("converged", rho_next)
                if iterations == self.options["maxiter"]:
                    return result("max_iterations", rho_next)
                rho = rho_next
                beta = ops.dot(residual, product) / denominator
            else:
                ops.axpy(-alpha, product, residual, scratch)
                rho_next = ops.dot(residual, residual)
                if not math.isfinite(rho_next):
                    return finish("non_finite")
                refresh = self.options["residual_refresh"]
                if (
                    math.sqrt(rho_next) <= tolerance
                    or (refresh and iterations % refresh == 0)
                    or iterations == self.options["maxiter"]
                ):
                    rho_next = true_residual()
                    if not math.isfinite(rho_next):
                        return result("non_finite", rho_next)
                    if math.sqrt(rho_next) <= tolerance:
                        return result("converged", rho_next)
                    if iterations == self.options["maxiter"]:
                        return result("max_iterations", rho_next)
                    # True residual replacement restarts the search direction.
                    ops.initialize_direction(residual, direction, False)
                    rho = rho_next
                    continue
                beta = rho_next / rho
            if not math.isfinite(beta):
                return finish("non_finite")
            ops.direction(beta, residual, direction, explicit)
            rho = rho_next
        rho = true_residual()
        if not math.isfinite(rho):
            return result("non_finite", rho)
        return result("converged" if math.sqrt(rho) <= tolerance else "max_iterations", rho)


def make_executor(name, arrays, options, scope, threads):
    if name == "native":
        return NativeExecutor(arrays, options, scope, threads)
    if name == "numpy-out":
        return ReferenceExecutor(NumpyOps(arrays), options, scope)
    if name == "torch-eager-out":
        return ReferenceExecutor(TorchOps(arrays, threads), options, scope)
    raise ValueError(f"unknown implementation: {name}")
