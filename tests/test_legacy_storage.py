import ctypes

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
