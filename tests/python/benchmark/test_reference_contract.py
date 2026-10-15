"""Reference recurrence accepts capabilities without a concrete backend type."""

import contextlib

import numpy as np
import pytest
from cgbench.reference.engine import ReferenceExecutor
from cgbench.reference.numpy import NumpyOps


class IndependentOperations:
    """A composed adapter with its own snapshot and execution-context policy."""

    def __init__(self, arrays):
        self.operations = NumpyOps(arrays)
        self.snapshots = 0
        self.context_entries = 0
        self.in_context = False

    def __getattr__(self, name):
        return getattr(self.operations, name)

    @property
    def initial_nbytes(self):
        return 123

    def snapshot_matrix(self):
        self.operations.snapshot_matrix()
        self.snapshots += 1

    @contextlib.contextmanager
    def execution_context(self):
        self.context_entries += 1
        self.in_context = True
        try:
            yield
        finally:
            self.in_context = False

    def matvec(self, vector, out):
        assert self.in_context
        self.operations.matvec(vector, out)


@pytest.mark.parametrize("scope", ("api", "prepared"))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
def test_reference_engine_uses_operation_capabilities(scope, preconditioner):
    matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
    rhs = np.array([1.0, 2.0])
    expected = np.linalg.solve(matrix, rhs)
    ops = IndependentOperations((matrix, rhs, np.zeros_like(rhs)))
    executor = ReferenceExecutor(
        ops,
        {
            "preconditioner": preconditioner,
            "algorithm": "standard",
            "rtol": 1e-10,
            "atol": 0.0,
            "maxiter": 100,
            "residual_refresh": 50,
        },
        scope,
    )
    assert ops.snapshots == int(scope == "prepared")
    if scope == "prepared":
        matrix.fill(np.nan)
    for _ in range(2):
        outcome = executor()
        assert outcome.converged
        np.testing.assert_allclose(outcome.x, expected)
        assert not ops.in_context
    assert ops.context_entries == 2
    assert executor.benchmark_metadata()["preconditioner_stored_bytes"] == (
        123 if preconditioner == "jacobi" else 0
    )
