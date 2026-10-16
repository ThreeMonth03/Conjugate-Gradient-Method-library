"""Result identity and convergence metadata contracts."""

import cgpy
import numpy as np
import pytest


def test_results_compare_by_identity_even_with_equal_or_shared_payloads():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    first = cgpy.solve(A, b)
    second = cgpy.solve(A, b)
    shared = cgpy.SolveResult(first.x, first.info)

    np.testing.assert_allclose(first.x, second.x)
    assert first == first
    assert first != second
    assert first != shared


def test_result_remains_a_dictionary_key_after_solution_mutation():
    result = cgpy.solve(np.eye(2), np.ones(2))
    labels = {result: "solution"}

    result.x.fill(0.0)

    assert labels[result] == "solution"


@pytest.mark.parametrize("preconditioner", ["none", "jacobi"])
def test_result_reports_search_direction_restarts_as_read_only(preconditioner):
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    result = cgpy.solve(
        A, b, preconditioner=preconditioner, residual_refresh=1, rtol=1e-10
    )

    assert result.converged
    assert result.restarts == result.info.restarts > 0
    assert result.restarts == result.iterations - 1
    np.testing.assert_allclose(result.x, np.linalg.solve(A, b), rtol=1e-9)
    with pytest.raises(AttributeError):
        result.restarts = 0
    assert result.restarts == result.info.restarts
