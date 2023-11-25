import sys
import os
# Get the absolute path of the directory containing the .so file
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../cpp"
# Add the directory to the Python path
sys.path.append(dir_path)
# Get the absolute path of the directory containing the utils.py
dir_path = os.path.dirname(os.path.abspath(__file__)) + "/../python"
# Add the directory to the Python path
sys.path.append(dir_path)

import pytest
import math
import _cgpy
import time
import numpy as np
import utils
from autograd import grad
import autograd.numpy as au

def test_linear_cg():
    a = np.random.rand(100, 100)
    a = utils.generate_pos_def(100)
    a = utils.generate_symmetric(100)
    while(np.linalg.cond(a) < 1000):
        a = np.random.rand(100, 100)
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


def test_nonlinear_cg():
    x = np.array([2., -1.8])

    for method in ["Fletcher_Reeves", "Polak_Ribiere", "Dai-Yuan", "Hager-Zhang"]:
        x, _ = utils.nonlinear_CG(x, 1e-7, 1e-4, 0.9, utils.nonlinear_func_1, utils.grad(utils.nonlinear_func_1), method)
        print('x',x)
        x = np.array(x)
        x_min = np.array([1, 1])
        print('x_min', x_min)
        assert(np.isclose(x_min, x).all())

    x = np.random.uniform(low=0.5, high=0.7, size=(100,))
    for method in ["Fletcher_Reeves", "Polak_Ribiere", "Dai-Yuan", "Hager-Zhang"]:
        x, _ = utils.nonlinear_CG(x, 1e-8, 1e-4, 0.9, utils.nonlinear_func_2, utils.grad(utils.nonlinear_func_2), method)
        print('x',x)
        x = np.array(x)
        x_min = np.zeros(100)
        print('x_min', x_min)
        assert(np.isclose(x_min, x).all())            

#if __name__ == '__main__':
    #est_linear_cg()
    #test_nonlinear_cg()