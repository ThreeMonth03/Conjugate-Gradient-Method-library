"""Preallocated eager PyTorch CPU operations for the reference recurrence."""


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

    def snapshot_matrix(self):
        self.matrix = self.matrix.clone()

    @property
    def initial_nbytes(self):
        return int(self.initial.numel() * self.initial.element_size())

    def execution_context(self):
        return self.torch.inference_mode()

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

    def outcome_x(self, x):
        return self.export(x)
