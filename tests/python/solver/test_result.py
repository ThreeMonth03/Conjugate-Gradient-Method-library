"""Result identity contracts for mutable solution arrays."""

import cgpy
import numpy as np


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
