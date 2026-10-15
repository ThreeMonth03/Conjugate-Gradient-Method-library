"""Preallocated NumPy vector operations for the reference recurrence."""

import contextlib


class NumpyOps:
    def __init__(self, arrays):
        import numpy as np

        self.np = np
        self.matrix, self.rhs, self.initial = arrays
        self.validate(True)

    def snapshot_matrix(self):
        self.matrix = self.matrix.copy()

    @property
    def initial_nbytes(self):
        return int(self.initial.nbytes)

    def execution_context(self):
        return contextlib.nullcontext()

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

    def outcome_x(self, x):
        return self.export(x)
