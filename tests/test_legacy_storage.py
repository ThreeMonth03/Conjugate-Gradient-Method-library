import ctypes

import numpy as np
import pytest
from _cgpy import CG
from _cgpy.Matrix import Accelerated_Matrix, Naive_Matrix


@pytest.mark.parametrize("matrix_type", [Naive_Matrix, Accelerated_Matrix])
def test_legacy_empty_matrix_roundtrip(matrix_type):
    empty = matrix_type(0, 3)
    copied = matrix_type(empty)
    assert (copied.nrow, copied.ncol) == (0, 3)
    assert copied.tolist() == []
    assert copied.norm() == 0

    rows = matrix_type([[], [], []])
    assert (rows.nrow, rows.ncol) == (3, 0)
    assert rows.tolist() == []
    assert rows.tolist2d() == [[], [], []]


@pytest.mark.parametrize("matrix_type", [Naive_Matrix, Accelerated_Matrix])
def test_legacy_rejects_shape_overflow(matrix_type):
    maximum = 2 ** (8 * ctypes.sizeof(ctypes.c_size_t)) - 1
    with pytest.raises(ValueError, match="element count"):
        matrix_type(maximum // 2 + 1, 2)
    with pytest.raises(ValueError, match="byte size"):
        matrix_type(maximum // ctypes.sizeof(ctypes.c_double) + 1, 1)
    with pytest.raises(ValueError, match="element count"):
        matrix_type(maximum // 2 + 1, 2, [])


@pytest.mark.parametrize("matrix_type", [Naive_Matrix, Accelerated_Matrix])
def test_legacy_matrix_copy_and_repr(matrix_type):
    original = matrix_type([[1.0, 2.0], [3.0, 4.0]])
    copied = matrix_type(original)
    original[0, 0] = 9.0
    assert copied[0, 0] == 1.0
    assert repr(copied) == f"{matrix_type.__name__}(2, 2)"


def test_legacy_solver_setters_accept_values():
    solver = CG.linear_CG()
    solver.set_epsilon(1e-9)
    solver.set_epoch(35)
    solver.set_number_of_threads(2)
    assert solver.get_epsilon() == 1e-9
    assert solver.get_epoch() == 35
    assert solver.get_number_of_threads() == 2


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize(
    "matrix,rhs,tolerance,status",
    [
        (np.eye(2), [1e-200, -1e-200], 1e-208, "non_finite"),
        (np.eye(2), [np.inf, 1.0], 1e-8, "non_finite"),
        (-np.eye(2), [1.0, 2.0], 1e-8, "non_positive_curvature"),
    ],
)
def test_legacy_solver_exposes_numerical_failure(accelerated, matrix, rhs, tolerance, status):
    matrix_type = Accelerated_Matrix if accelerated else Naive_Matrix
    solver = CG.linear_CG(tolerance, 20, 1)
    solve = solver.solve_by_Accelerated_Matrix if accelerated else solver.solve_by_Naive_Matrix
    initial = matrix_type([0.0, 0.0])
    with pytest.raises(RuntimeError, match=status):
        solve(matrix_type(matrix), matrix_type(rhs), initial)
    np.testing.assert_array_equal(initial.tolist(), [0.0, 0.0])


@pytest.mark.parametrize("invalid", [0, -1])
def test_legacy_thread_setters_reject_invalid_values_without_changing_state(invalid):
    matrix = Accelerated_Matrix([[1.0, 0.0], [0.0, 1.0]])
    matrix.set_number_of_threads(2)
    solver = CG.linear_CG()
    solver.set_number_of_threads(2)
    for target in (matrix, solver):
        with pytest.raises(ValueError, match="positive"):
            target.set_number_of_threads(invalid)
        assert target.get_number_of_threads() == 2
    assert (matrix * 2.0).tolist2d() == [[2.0, 0.0], [0.0, 2.0]]
