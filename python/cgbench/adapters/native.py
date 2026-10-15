"""Native CPU solver API adapter with explicit prepared and API scopes."""

import time

from .. import execution


class NativeExecutor:
    def __init__(self, arrays, options, scope, threads):
        import cgpy
        import numpy as np

        self.np = np
        self.cgpy = cgpy
        self.matrix, self.rhs, self.initial = arrays
        if hasattr(self.matrix, "indptr"):
            self.matrix = cgpy.CsrMatrix(
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
            plan.preconditioner_setup_ns
            if self.preconditioner == "jacobi"
            else 0
        )
        self.setup_metadata["preconditioner_stored_bytes"] = (
            plan.preconditioner_nbytes
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
        return execution.Outcome(
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
