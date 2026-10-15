"""Preallocated NumPy/PyTorch references and the public native API adapter."""

import dataclasses
import math
import sys
import time

from . import numerics

CUDA_IMPLEMENTATIONS = ("cuda-native", "cuda-cublas", "torch-cuda-eager")


@dataclasses.dataclass
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
        if hasattr(self.matrix, "indptr"):
            self.matrix = cgpy.CSRMatrix(
                self.matrix.indptr,
                self.matrix.indices,
                self.matrix.data,
                self.matrix.shape,
            )
        self.options = dict(options)
        self.preconditioner = self.options.pop("preconditioner", "none")
        self.dense_kernel = self.options.pop("dense_kernel", "native")
        if self.dense_kernel == "blas" and not cgpy.blas_enabled():
            raise ImportError("native-blas requires a BLAS-enabled cgpy build")
        self.scope = scope
        self.threads = threads
        self.setup_metadata = {"preconditioner": self.preconditioner}
        settings = {
            "preconditioner": self.preconditioner,
            "dense_kernel": self.dense_kernel,
        }
        if scope == "prepared":
            start = time.perf_counter_ns()
            self.solver = cgpy.prepare(
                self.matrix, num_threads=threads, **settings
            )
            self.setup_metadata["operator_setup_elapsed_ns"] = (
                time.perf_counter_ns() - start
            )
            self.workspace = self.solver.create_workspace()
            self.x = np.empty_like(self.initial)
            plan = self.solver
        elif self.preconditioner != "none":
            plan = cgpy.prepare(self.matrix, num_threads=threads, **settings)
        else:
            plan = None
        self.setup_metadata["preconditioner_setup_elapsed_ns"] = (
            getattr(plan, "preconditioner_setup_ns", None)
            if self.preconditioner == "jacobi"
            else 0
        )
        self.setup_metadata["preconditioner_stored_bytes"] = (
            int(getattr(plan, "preconditioner_nbytes", self.initial.nbytes))
            if self.preconditioner == "jacobi"
            else 0
        )

    def __call__(self):
        if self.scope == "prepared":
            # Reset is part of EVERY measured invocation, including repetitions.
            self.np.copyto(self.x, self.initial)
            info = self.solver.solve_into(
                self.rhs, self.x, workspace=self.workspace, **self.options
            )
            x = self.x
        else:
            settings = {
                "preconditioner": self.preconditioner,
                "dense_kernel": self.dense_kernel,
            }
            info = self.cgpy.solve(
                self.matrix,
                self.rhs,
                x0=self.initial,
                num_threads=self.threads,
                **settings,
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

    def benchmark_metadata(self):
        return {
            **self.setup_metadata,
            "implementation_role": "native",
            "dense_kernel": self.dense_kernel,
            "blas": self.cgpy.build_info()["blas"]
            if self.dense_kernel == "blas"
            else None,
            "solver_family": "pcg"
            if self.preconditioner == "jacobi"
            else "cg",
            "setup_in_timed_call": self.scope == "api",
            "setup_measurement": "native plan setup outside timing; API setup repeats inside every timed solve",
            "residual_policy": self.options["algorithm"]
            + " with residual_refresh="
            + str(self.options["residual_refresh"]),
        }


class NumpyOps:
    def __init__(self, arrays):
        import numpy as np

        self.np = np
        self.matrix, self.rhs, self.initial = arrays
        self.validate(True)

    def workspace(self):
        return [self.empty_vector() for _ in range(5)]

    def empty_vector(self):
        return self.np.empty_like(self.initial)

    def inverse_diagonal(self):
        diagonal = self.matrix.diagonal().copy()
        if not self.np.all(self.np.isfinite(diagonal)) or not self.np.all(
            diagonal > 0.0
        ):
            raise ValueError("Jacobi requires a finite positive diagonal")
        self.np.reciprocal(diagonal, out=diagonal)
        if not self.np.all(self.np.isfinite(diagonal)) or not self.np.all(
            diagonal > 0.0
        ):
            raise ValueError("Jacobi reciprocal must be finite and positive")
        return diagonal

    def precondition(self, inverse, residual, out):
        self.np.multiply(inverse, residual, out=out)

    def validate(self, api):
        arrays = (
            (self.matrix, self.rhs, self.initial)
            if api
            else (self.rhs, self.initial)
        )
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
        matrix, rhs, initial = arrays
        self.original_matrix = matrix
        self.sparse = hasattr(matrix, "indptr")
        if self.sparse:
            self.matrix = torch.sparse_csr_tensor(
                torch.from_numpy(matrix.indptr),
                torch.from_numpy(matrix.indices),
                torch.from_numpy(matrix.data),
                size=matrix.shape,
            )
        else:
            self.matrix = torch.from_numpy(matrix)
        self.rhs, self.initial = (
            torch.from_numpy(rhs),
            torch.from_numpy(initial),
        )
        self.validate(True)
        if self.sparse and not bool(
            torch.isfinite(self.matrix.values()).all()
        ):
            raise ValueError("CSR inputs must be finite")

    def workspace(self):
        return [self.empty_vector() for _ in range(5)]

    def empty_vector(self):
        return self.torch.empty_like(self.initial)

    def _diagonal(self):
        return self.torch.from_numpy(self.original_matrix.diagonal().copy())

    def inverse_diagonal(self):
        diagonal = self._diagonal()
        if not bool(self.torch.isfinite(diagonal).all()) or not bool(
            (diagonal > 0).all()
        ):
            raise ValueError("Jacobi requires a finite positive diagonal")
        diagonal.reciprocal_()
        if not bool(self.torch.isfinite(diagonal).all()) or not bool(
            (diagonal > 0).all()
        ):
            raise ValueError("Jacobi reciprocal must be finite and positive")
        return diagonal

    def precondition(self, inverse, residual, out):
        self.torch.mul(inverse, residual, out=out)

    def validate(self, api):
        arrays = (
            (self.matrix, self.rhs, self.initial)
            if api and not self.sparse
            else (self.rhs, self.initial)
        )
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


class ScipyOps(NumpyOps):
    """Public CSR matvec with preallocated NumPy vector updates.

    SciPy's public ``A @ x`` allocates a product. Keep that cost visible rather
    than depending on private sparsetools or claiming an allocation-free solve.
    """

    def __init__(self, arrays):
        super().__init__(arrays)
        if not hasattr(self.matrix, "indptr") or self.matrix.format != "csr":
            raise ValueError("scipy-out requires canonical CSR inputs")
        if not self.matrix.has_canonical_format or not self.np.all(
            self.np.isfinite(self.matrix.data)
        ):
            raise ValueError("CSR inputs must be canonical and finite")

    def validate(self, api):
        # All implementations receive a prepared, validated CSR object. Input
        # conversion and canonical validation belong outside both solve scopes.
        super().validate(False)

    def matvec(self, vector, out):
        self.np.copyto(out, self.matrix @ vector)


class ReferenceExecutor:
    """Use the native solver's recurrence, restart, and true-residual policy."""

    def __init__(self, ops, options, scope):
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
            self.ops.matrix = (
                self.ops.matrix.copy()
                if isinstance(ops, NumpyOps)
                else self.ops.matrix.clone()
            )
        self.storage = ops.workspace() if scope == "prepared" else None
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
            "preconditioner_stored_bytes": int(self.ops.initial.numel() * 8)
            if isinstance(self.ops, TorchOps)
            and self.preconditioner == "jacobi"
            else int(self.ops.initial.size * 8)
            if self.preconditioner == "jacobi"
            else 0,
            "setup_in_timed_call": self.scope == "api",
            "setup_measurement": "single untimed setup probe",
            "residual_policy": self.options["algorithm"]
            + " with residual_refresh="
            + str(self.options["residual_refresh"]),
        }

    def __call__(self):
        ops = self.ops
        if isinstance(ops, TorchOps):
            with ops.torch.inference_mode():
                return self._solve()
        return self._solve()

    def _solve(self):
        ops = self.ops
        storage = self.storage if self.storage is not None else ops.workspace()
        inverse = self.inverse
        if self.preconditioner == "jacobi" and self.scope == "api":
            inverse = ops.inverse_diagonal()
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
            output = (
                ops.outcome_x(x)
                if hasattr(ops, "outcome_x")
                else ops.export(x)
            )
            return Outcome(output, status, iterations, matvecs, residual_norm)

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
                refresh = self.options["residual_refresh"]
                if (
                    norm_next <= tolerance
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
                    continue
                if (
                    not math.isfinite(rho_next)
                    or rho_next < sys.float_info.min
                ):
                    return finish("non_finite")
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


def make_executor(
    name, arrays, options, scope, threads, *, device="cpu", cuda_device=0
):
    if name in CUDA_IMPLEMENTATIONS:
        if device != "cuda":
            raise ValueError("CUDA candidates require --device cuda")
        from . import cuda_backends

        if name == "torch-cuda-eager":
            return cuda_backends.TorchCudaExecutor(
                arrays, options, scope, threads, cuda_device
            )
        return cuda_backends.CudaNativeExecutor(
            arrays,
            options,
            scope,
            cuda_device,
            "cublas" if name == "cuda-cublas" else "native",
        )
    if device != "cpu":
        raise ValueError(
            "CPU candidates require --device cpu; CUDA never falls back to CPU"
        )
    if name == "native-blas":
        if hasattr(arrays[0], "indptr"):
            raise ValueError("native-blas requires dense inputs")
        return NativeExecutor(
            arrays, {**options, "dense_kernel": "blas"}, scope, threads
        )
    if name in ("scipy-lu-total", "scipy-lu-reuse"):
        from . import sparse_solvers

        return sparse_solvers.SparseLUExecutor(
            arrays, options, scope, reuse=name.endswith("reuse")
        )
    if name in ("scipy-cg", "scipy-pcg"):
        from . import sparse_solvers

        return sparse_solvers.ScipyKrylovExecutor(
            arrays, options, scope, jacobi=name == "scipy-pcg"
        )
    aliases = {
        "native-cg": ("native", "none"),
        "native-pcg": ("native", "jacobi"),
        "scipy-reference-cg": ("scipy-out", "none"),
        "scipy-reference-pcg": ("scipy-out", "jacobi"),
        "torch-cg": ("torch-eager-out", "none"),
        "torch-pcg": ("torch-eager-out", "jacobi"),
    }
    if name in aliases:
        name, preconditioner = aliases[name]
        options = {**options, "preconditioner": preconditioner}
    if name == "native":
        return NativeExecutor(
            arrays, {**options, "dense_kernel": "native"}, scope, threads
        )
    if name == "numpy-out":
        if hasattr(arrays[0], "indptr"):
            raise ValueError(
                "numpy-out requires dense inputs; use scipy-out for CSR"
            )
        return ReferenceExecutor(NumpyOps(arrays), options, scope)
    if name == "scipy-out":
        return ReferenceExecutor(ScipyOps(arrays), options, scope)
    if name == "torch-eager-out":
        return ReferenceExecutor(TorchOps(arrays, threads), options, scope)
    raise ValueError(f"unknown implementation: {name}")
