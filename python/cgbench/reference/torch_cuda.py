"""Eager float64 CUDA operations for the independent reference recurrence."""

import time

from .. import cuda_runtime
from . import torch


class TorchCudaOps(torch.TorchOps):
    """FP64 eager operations on one explicit CUDA device and default stream."""

    def __init__(self, arrays, threads, device):
        torch = cuda_runtime.torch_cuda(device)
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
