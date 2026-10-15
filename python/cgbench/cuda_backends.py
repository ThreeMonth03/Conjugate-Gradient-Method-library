"""Explicit device-resident CUDA adapters; imports never initialize CUDA early."""

import time

from . import backends


def native_cuda(device):
    """Require the requested native CUDA device, with no CPU fallback."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    try:
        from cgpy import cuda
    except ImportError as exc:
        raise RuntimeError(
            "CUDA requested but the native CUDA extension is unavailable"
        ) from exc
    if not cuda.enabled() or not cuda.is_available(device):
        raise RuntimeError(
            f"CUDA requested but native CUDA device {device} is unavailable"
        )
    return cuda


def torch_cuda(device):
    """Require CUDA PyTorch and a valid logical device ordinal."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "CUDA requested but PyTorch is unavailable"
        ) from exc
    if not torch.cuda.is_available() or device >= torch.cuda.device_count():
        raise RuntimeError(
            f"CUDA requested but PyTorch CUDA device {device} is unavailable"
        )
    return torch


def preflight_cuda(names, device):
    """Fail before collecting or writing timings if any requested GPU is absent."""
    if device < 0:
        raise ValueError("CUDA device must be nonnegative")
    if not names or any(
        name not in backends.CUDA_IMPLEMENTATIONS for name in names
    ):
        raise ValueError("CUDA comparison requires explicit CUDA candidates")
    if any(name in ("cuda-native", "cuda-cublas") for name in names):
        native_cuda(device)
    if "torch-cuda-eager" in names:
        torch_cuda(device)


def _host_outcome(outcome, x):
    return backends.Outcome(
        x,
        outcome.status,
        outcome.iterations,
        outcome.matvecs,
        outcome.residual_norm,
    )


class CudaNativeExecutor:
    """Prepared and one-shot solves over owning device matrices/vectors.

    Input upload/snapshot and correctness download are outside the wall timer.
    Every invocation charges its x0 device copy. API scope additionally charges
    output/workspace allocation and per-call preconditioner setup.
    """

    def __init__(self, arrays, options, scope, device, kernel):
        if scope not in ("prepared", "api"):
            raise ValueError(
                "CUDA supports prepared/api device-resident scopes"
            )
        if kernel == "cublas" and hasattr(arrays[0], "indptr"):
            raise ValueError("cuda-cublas requires dense inputs")
        self.cuda = native_cuda(device)
        self.device, self.kernel, self.scope = device, kernel, scope
        self.options = dict(options)
        self.preconditioner = self.options.pop("preconditioner", "none")
        self.options.pop("dense_kernel", None)
        self.synchronize()
        start = time.perf_counter_ns()
        matrix, rhs, initial = arrays
        if hasattr(matrix, "indptr"):
            import cgpy

            host_matrix = cgpy.CSRMatrix(
                matrix.indptr, matrix.indices, matrix.data, matrix.shape
            )
            self.matrix = self.cuda.CSRMatrix.from_host(
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
            self.workspace = plan.create_workspace(stream=0)
            self.x = plan.create_vector(stream=0)
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
        else:
            info = self.cuda.solve(
                self.matrix,
                self.rhs,
                x0=self.initial,
                device=self.device,
                stream=0,
                dense_kernel=self.kernel,
                preconditioner=self.preconditioner,
                **self.options,
            )
            x = info.x
        return backends.Outcome(
            x,
            str(info.status),
            info.iterations,
            info.matvecs,
            info.residual_norm,
        )

    def host_outcome(self, outcome):
        return _host_outcome(outcome, outcome.x.to_host(stream=0))

    def benchmark_metadata(self):
        return {
            "implementation_role": "native",
            "device": "cuda",
            "cuda_device": self.device,
            "stream": 0,
            "dense_kernel": self.kernel,
            "fused": self.options["fused"],
            "preconditioner": self.preconditioner,
            "preconditioner_setup_elapsed_ns": self.preconditioner_setup_ns,
            "preconditioner_stored_bytes": self.preconditioner_nbytes,
            "host_upload_setup_elapsed_ns": self.upload_ns,
            "resident_adapter_setup_elapsed_ns": self.setup_ns,
            "setup_in_timed_call": self.scope == "api",
            "setup_measurement": "single untimed completed GPU setup probe",
            "setup_probe_scope": "operator snapshot and Jacobi probe; reusable output/workspace only for prepared scope",
            "api_setup_scope": "output/workspace allocations and per-call Jacobi; matrix upload/snapshot excluded",
            "transfer_scope": "input uploads and correctness output downloads excluded",
            "reset_scope": "device x0 copy included in every invocation",
            "timing": "wall completion; stream fence before block and after last invocation",
            "control_scope": "synchronous native solve; host dispatch and scalar checks included",
            "residual_policy": self.options["algorithm"]
            + " with residual_refresh="
            + str(self.options["residual_refresh"]),
        }


class TorchCudaOps(backends.TorchOps):
    """FP64 eager operations on one explicit CUDA device and default stream."""

    def __init__(self, arrays, threads, device):
        torch = torch_cuda(device)
        super().__init__(arrays, threads)
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(self.device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.synchronize()
        start = time.perf_counter_ns()
        self.matrix = self.matrix.to(self.device)
        self.rhs = self.rhs.to(self.device)
        self.initial = self.initial.to(self.device)
        self.diagonal_positions = None
        if self.sparse:
            # Structural CSR lookup is prepared outside timing, like canonical
            # index validation. Values/reciprocal are extracted on GPU per M setup.
            matrix = arrays[0]
            positions = []
            for row in range(matrix.shape[0]):
                start_row, end_row = matrix.indptr[row : row + 2]
                matches = (matrix.indices[start_row:end_row] == row).nonzero()[
                    0
                ]
                if len(matches) != 1:
                    raise ValueError(
                        "Jacobi-compatible CSR requires one diagonal per row"
                    )
                positions.append(int(start_row + matches[0]))
            self.diagonal_positions = torch.tensor(
                positions, dtype=torch.int64, device=self.device
            )
        self.synchronize()
        self.upload_ns = time.perf_counter_ns() - start

    def synchronize(self):
        self.torch.cuda.synchronize(self.device)

    def _diagonal(self):
        return (
            self.matrix.values().index_select(0, self.diagonal_positions)
            if self.sparse
            else self.matrix.diagonal().clone()
        )

    @staticmethod
    def outcome_x(x):
        return x

    @staticmethod
    def export(x):
        # Used only by the numerical-range norm fallback. Ordinary solve output
        # stays on GPU and is downloaded exclusively by host_outcome below.
        return x.detach().cpu().numpy()


class TorchCudaExecutor(backends.ReferenceExecutor):
    def __init__(self, arrays, options, scope, threads, device):
        if scope not in ("prepared", "api"):
            raise ValueError(
                "CUDA supports prepared/api device-resident scopes"
            )
        ops = TorchCudaOps(arrays, threads, device)
        start = time.perf_counter_ns()
        super().__init__(ops, options, scope)
        self.synchronize()
        self.setup_ns = time.perf_counter_ns() - start

    def synchronize(self):
        self.ops.synchronize()

    def host_outcome(self, outcome):
        return _host_outcome(outcome, self.ops.export(outcome.x).copy())

    def benchmark_metadata(self):
        return {
            **super().benchmark_metadata(),
            "device": "cuda",
            "cuda_device": self.ops.device.index,
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
