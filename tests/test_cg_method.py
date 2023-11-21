import pytest
import math
import _cgpy
import time
import numpy as np
import utils

def test_linear_cg():
    a =  np.random.rand(100, 100)
    a = utils.generate_pos_def(100)
    a = utils.generate_symmetric(100)
    b = np.random.rand(100)
    x = np.random.rand(100)
    x = utils.linear_CG(x, a, b, 5e-7)
    print('x',x)
    x_min = np.linalg.solve(a, b) # Differentiate to find the minimizer
    print('x_min', x_min)
    assert(np.isclose(x_min, x).all())
    mat_a = _cgpy.Matrix(a)
    mat_b = _cgpy.Matrix(b)
    mat_x = _cgpy.Matrix(x)
    linear_cg_model = _cgpy.linear_CG(mat_a, mat_b, mat_x)
    mat_x_min = linear_cg_model.solve()
    np_mat_x_min = np.array(mat_x_min.tolist())
    print('np_mat_x_min', np_mat_x_min)
    assert(np.isclose(x_min, np_mat_x_min).all())

if __name__ == '__main__':
    test_linear_cg()