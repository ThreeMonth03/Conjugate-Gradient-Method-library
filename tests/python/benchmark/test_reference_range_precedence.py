"""Keep recurrence range failure ahead of checkpoint recovery, like native CG."""

import math
import sys

import numpy as np
import pytest
from cgbench.reference.engine import ReferenceExecutor
from cgbench.reference.numpy import NumpyOps


class RecurrenceRangeOperations(NumpyOps):
    """Inject a range-limited recurrence while retaining real true residuals.

    Deliberate recurrence drift isolates exit precedence from a particular
    BLAS implementation's rounding. Matvecs, norms and true residuals still
    use ordinary NumPy operations on the supplied finite SPD system.
    """

    def __init__(self, arrays, recurrence_value):
        super().__init__(arrays)
        self.recurrence_value = recurrence_value
        self.residual_vector = None
        self.residual_evaluations = 0
        self.direction_initializations = 0
        self.precondition_calls = 0
        self.recurrence_updates = 0
        self.recurrence_square = None
        self.recurrence_square_pending = False

    def residual(self, x, residual, product, explicit):
        self.residual_vector = residual
        self.residual_evaluations += 1
        self.recurrence_square_pending = False
        super().residual(x, residual, product, explicit)

    def axpy(self, alpha, source, destination, scratch):
        super().axpy(alpha, source, destination, scratch)
        if destination is self.residual_vector:
            destination.fill(self.recurrence_value)
            self.recurrence_updates += 1
            self.recurrence_square_pending = True

    def dot(self, left, right):
        value = super().dot(left, right)
        if (
            self.recurrence_square_pending
            and left is self.residual_vector
            and right is self.residual_vector
        ):
            self.recurrence_square = value
            self.recurrence_square_pending = False
        return value

    def initialize_direction(self, residual, direction, explicit):
        self.direction_initializations += 1
        super().initialize_direction(residual, direction, explicit)

    def precondition(self, inverse, residual, out):
        self.precondition_calls += 1
        super().precondition(inverse, residual, out)


def run_range_case(
    matrix,
    recurrence_value,
    *,
    preconditioner,
    scope="prepared",
    maxiter=3,
    residual_refresh=1,
    atol=0.0,
):
    rhs = np.array([1.0, 2.0])
    ops = RecurrenceRangeOperations(
        (matrix, rhs, np.zeros_like(rhs)), recurrence_value
    )
    execute = ReferenceExecutor(
        ops,
        {
            "algorithm": "standard",
            "preconditioner": preconditioner,
            "rtol": 0.0,
            "atol": atol,
            "maxiter": maxiter,
            "residual_refresh": residual_refresh,
        },
        scope,
    )
    with np.errstate(over="ignore", under="ignore"):
        outcome = execute()
    # Exactly one update, followed by the mandatory final true residual. A
    # refresh recovery must not initialize a second direction or apply M again.
    assert (outcome.iterations, outcome.matvecs, outcome.restarts) == (1, 3, 0)
    assert ops.residual_evaluations == 2
    assert ops.recurrence_updates == 1
    assert ops.direction_initializations == 1
    assert ops.precondition_calls == int(preconditioner == "jacobi")
    assert math.isfinite(outcome.residual_norm)
    expected_norm = math.hypot(*(rhs - matrix @ outcome.x))
    assert outcome.residual_norm == pytest.approx(expected_norm)
    if abs(recurrence_value) < 1.0:
        assert 0.0 <= ops.recurrence_square < sys.float_info.min
    else:
        assert math.isinf(ops.recurrence_square)
    return outcome


@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
@pytest.mark.parametrize("recurrence_value", (1e-200, 1e-160, 1e200))
@pytest.mark.parametrize(
    ("maxiter", "residual_refresh"),
    ((3, 1), (1, 0)),
    ids=("refresh", "iteration-limit"),
)
def test_nonconverged_unusable_square_precedes_refresh_or_iteration_limit(
    preconditioner, recurrence_value, maxiter, residual_refresh
):
    outcome = run_range_case(
        np.array([[4.0, 1.0], [1.0, 3.0]]),
        recurrence_value,
        preconditioner=preconditioner,
        maxiter=maxiter,
        residual_refresh=residual_refresh,
    )
    assert outcome.status == "non_finite"
    assert outcome.residual_norm > 0.0


@pytest.mark.parametrize("scope", ("api", "prepared", "session"))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
@pytest.mark.parametrize("recurrence_value", (1e-200, 1e200))
def test_zero_true_residual_does_not_override_recurrence_range_failure(
    scope, preconditioner, recurrence_value
):
    outcome = run_range_case(
        np.eye(2),
        recurrence_value,
        preconditioner=preconditioner,
        scope=scope,
    )
    assert outcome.status == "non_finite"
    assert outcome.residual_norm == 0.0


@pytest.mark.parametrize("scope", ("api", "prepared", "session"))
@pytest.mark.parametrize("preconditioner", ("none", "jacobi"))
@pytest.mark.parametrize("recurrence_value", (1e-200, 1e-160))
def test_converged_candidate_with_underflowed_square_still_gets_verified(
    scope, preconditioner, recurrence_value
):
    outcome = run_range_case(
        np.eye(2),
        recurrence_value,
        preconditioner=preconditioner,
        scope=scope,
        atol=1e-150,
    )
    assert outcome.status == "converged"
    assert outcome.residual_norm == 0.0
