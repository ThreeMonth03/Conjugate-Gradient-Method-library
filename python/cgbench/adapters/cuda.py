"""Completed device-resident native and PyTorch benchmark adapters."""

import time

from .. import cuda_runtime, execution, validation
from ..reference import engine, torch_cuda


def _host_outcome(outcome, x):
    return execution.Outcome(
        x,
        outcome.status,
        outcome.iterations,
        outcome.matvecs,
        outcome.residual_norm,
        outcome.restarts,
    )


class CudaNativeExecutor:
    """Prepared and one-shot solves over owning device matrices/vectors.

    Input upload/snapshot and correctness download are outside the wall timer.
    Every invocation charges its x0 device copy. API scope additionally charges
    output/workspace allocation and per-call preconditioner setup.
    """

    def __init__(self, arrays, options, scope, device, kernel):
        validation.validate_scope(scope, "cuda")
        if kernel == "cublas" and hasattr(arrays[0], "indptr"):
            raise ValueError("cuda-cublas requires dense inputs")
        self.cuda = cuda_runtime.native_cuda(device)
        self.device, self.kernel, self.scope = device, kernel, scope
        self.options = dict(options)
        self.execution_mode = validation.validate_execution_mode(
            self.options, "cuda"
        )
        self.options.pop("execution_mode", None)
        self.preconditioner = self.options.pop("preconditioner", "none")
        self.options.pop("dense_kernel", None)
        self.synchronize()
        start = time.perf_counter_ns()
        matrix, rhs, initial = arrays
        if hasattr(matrix, "indptr"):
            import cgpy

            host_matrix = cgpy.CsrMatrix(
                matrix.indptr, matrix.indices, matrix.data, matrix.shape
            )
            self.matrix = self.cuda.CsrMatrix.from_host(
                host_matrix, device=device, stream=0
            )
        else:
            self.matrix = self.cuda.DenseMatrix.from_host(
                matrix, device=device, stream=0
            )
        self.rhs = self.cuda.Vector.from_host(rhs, device=device, stream=0)
        self.initial = self.cuda.Vector.from_host(
            initial, device=device, stream=0
        )
        self.synchronize()
        self.upload_ns = time.perf_counter_ns() - start
        start = time.perf_counter_ns()
        # This probe also discloses M setup for API scope, which repeats setup
        # inside every one-shot call rather than reusing this prepared plan.
        plan = self.cuda.prepare(
            self.matrix,
            device=device,
            stream=0,
            dense_kernel=kernel,
            preconditioner=self.preconditioner,
        )
        self.preconditioner_setup_ns = plan.preconditioner_setup_ns
        self.preconditioner_nbytes = plan.preconditioner_nbytes
        if scope == "prepared":
            self.plan = plan
            self.workspace = plan.create_workspace(
                stream=0, execution_mode=self.execution_mode
            )
            self.x = plan.create_vector(stream=0)
        elif scope == "session":
            self.synchronize()
            session_start = time.perf_counter_ns()
            self.session = self.cuda.Session(
                rhs.size,
                device=device,
                stream=0,
                dense_kernel=kernel,
                execution_mode=self.execution_mode,
            )
            self.synchronize()
            self.session_setup_ns = time.perf_counter_ns() - session_start
        self.synchronize()
        self.setup_ns = time.perf_counter_ns() - start

    def synchronize(self):
        self.cuda.synchronize(device=self.device, stream=0)

    def __call__(self):
        if self.scope == "prepared":
            self.x.copy_from_device(self.initial, stream=0)
            info = self.plan.solve_into(
                self.rhs, self.x, workspace=self.workspace, **self.options
            )
            x = self.x
        elif self.scope == "session":
            info = self.session.solve(
                self.matrix,
                self.rhs,
                x0=self.initial,
                preconditioner=self.preconditioner,
                **self.options,
            )
            x = info.x
        else:
            info = self.cuda.solve(
                self.matrix,
                self.rhs,
                x0=self.initial,
                device=self.device,
                stream=0,
                dense_kernel=self.kernel,
                preconditioner=self.preconditioner,
                execution_mode=self.execution_mode,
                **self.options,
            )
            x = info.x
        return execution.Outcome(
            x,
            str(info.status),
            info.iterations,
            info.matvecs,
            info.residual_norm,
            info.restarts,
        )

    def host_outcome(self, outcome):
        return _host_outcome(outcome, outcome.x.to_host(stream=0))

    def benchmark_metadata(self):
        metadata = {
            "implementation_role": "native",
            "device": "cuda",
            "cuda_device": self.device,
            "stream": 0,
            "dense_kernel": self.kernel,
            "execution_mode": self.execution_mode,
            "fused": self.options["fused"],
            "preconditioner": self.preconditioner,
            "preconditioner_setup_elapsed_ns": self.preconditioner_setup_ns,
            "preconditioner_stored_bytes": self.preconditioner_nbytes,
            "host_upload_setup_elapsed_ns": self.upload_ns,
            "resident_adapter_setup_elapsed_ns": self.setup_ns,
            "setup_in_timed_call": self.scope in ("api", "session"),
            "setup_measurement": "single untimed completed GPU setup probe",
            "setup_probe_scope": "operator snapshot and Jacobi probe; reusable output/workspace only for prepared scope",
            "api_setup_scope": "output/workspace allocations and per-call Jacobi; matrix upload/snapshot excluded",
            "transfer_scope": "input uploads and correctness output downloads excluded",
            "reset_scope": "device x0 copy included in every invocation",
            "timing": "wall completion; stream fence before block and after last invocation",
            "control_scope": (
                "synchronous native solve; device recurrence scalars, completed status packets and true-residual checks included"
                if self.execution_mode == "device"
                else "synchronous native solve; host dispatch and scalar checks included"
            ),
            "residual_policy": self.options["algorithm"]
            + " with residual_refresh="
            + str(self.options["residual_refresh"]),
        }
        if self.scope == "session":
            metadata.update(
                session_construction_elapsed_ns=self.session_setup_ns,
                workspace_reused=True,
                workspace_in_timed_call=False,
                output_reused=False,
                output_in_timed_call=True,
                preconditioner_reused=False,
                preconditioner_in_timed_call=self.preconditioner == "jacobi",
                api_setup_scope="fresh output and per-call Jacobi; session workspace reused; matrix upload excluded",
                setup_probe_scope="untimed operator snapshot/Jacobi probe and completed Session construction; no retained matrix/M cache",
            )
        return metadata


class TorchCudaExecutor(engine.ReferenceExecutor):
    def __init__(self, arrays, options, scope, threads, device):
        validation.validate_scope(scope, "cuda")
        self.native_execution_mode = validation.validate_execution_mode(
            options, "cuda"
        )
        options = {
            key: value
            for key, value in options.items()
            if key != "execution_mode"
        }
        ops = torch_cuda.TorchCudaOps(arrays, threads, device)
        start = time.perf_counter_ns()
        super().__init__(ops, options, scope)
        self.synchronize()
        self.setup_ns = time.perf_counter_ns() - start

    def synchronize(self):
        self.ops.synchronize()

    def host_outcome(self, outcome):
        return _host_outcome(outcome, self.ops.export(outcome.x).copy())

    def benchmark_metadata(self):
        metadata = {
            **super().benchmark_metadata(),
            "device": "cuda",
            "cuda_device": self.ops.device.index,
            "native_execution_mode": self.native_execution_mode,
            "native_execution_mode_applies": False,
            "stream": int(
                self.ops.torch.cuda.current_stream(self.ops.device).cuda_stream
            ),
            "host_upload_setup_elapsed_ns": self.ops.upload_ns,
            "resident_adapter_setup_elapsed_ns": self.setup_ns,
            "setup_probe_scope": "prepared snapshot/workspace when selected; Jacobi setup separately recorded",
            "api_setup_scope": "output/workspace tensor allocations and per-call Jacobi; matrix upload excluded",
            "transfer_scope": "input uploads and correctness output downloads excluded",
            "reset_scope": "device x0 copy included in every invocation",
            "timing": "wall completion; device fence before block and after last invocation",
            "control_scope": "CPU eager CG control and GPU scalar synchronizations included; numerical-range fallback may download residual",
            "precision": "float64; TF32 disabled; no mixed precision",
            "workspace_policy": "preallocated prepared tensors; API uses PyTorch caching allocator",
        }
        if self.scope == "session":
            metadata.update(
                workspace_reused=True,
                workspace_in_timed_call=False,
                output_reused=False,
                output_in_timed_call=True,
                preconditioner_reused=False,
                preconditioner_in_timed_call=self.preconditioner == "jacobi",
                api_setup_scope="fresh output tensor and per-call Jacobi; session workvectors reused; matrix upload excluded",
                workspace_policy="cached iteration vectors; fresh output/inverse per call; PyTorch caching allocator",
                setup_probe_scope="untimed iteration-vector allocation and Jacobi probe; no matrix/M cache",
            )
        return metadata
