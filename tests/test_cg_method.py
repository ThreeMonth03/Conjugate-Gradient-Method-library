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
import time
import numpy as np
import utils
from autograd import grad
import autograd.numpy as au

from _cgpy import CG
from _cgpy.Matrix import Naive_Matrix
from _cgpy.Matrix import Accelerated_Matrix


def test_linear_cg():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(23, 23))
    a = q @ q.T + np.eye(23)
    b = rng.normal(size=23)
    x0 = rng.normal(size=23)
    expected = np.linalg.solve(a, b)
    implementations = [
        utils.np_linear_CG(x0.copy(), a, b, 1e-10, epoch=1000),
        utils.custom_linear_CG(
            x=x0.copy(), a=a, b=b, epsilon=1e-10, epoch=1000, num_threads=1
        ),
        utils.custom_linear_CG(
            x=x0.copy(), a=a, b=b, epsilon=1e-10, epoch=1000, num_threads=2
        ),
    ]
    for actual in implementations:
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-9)
        assert np.linalg.norm(b - a @ actual) <= 1e-9


def test_nonlinear_cg():
    np.random.seed(3)
    x_rand = np.random.uniform(low=3, high=5, size=(2,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        x = np.copy(x_rand)
        print("case 1 np: ", method)
        x, _ = utils.np_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

        print("case 1 custom : ", method, " , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
            1,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

        print("case 1 custom : ", method, " , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            5e-8,
            0.5,
            0.8,
            utils.nonlinear_func_1,
            utils.grad(utils.nonlinear_func_1),
            method,
        )
        x = np.array(x)
        x_min = np.array([1, 1])
        assert np.isclose(x_min, x).all()

    x_rand = np.random.uniform(low=0.5, high=0.7, size=(100,))
    for method in ["Fletcher_Reeves", "Dai-Yuan", "Hager-Zhang"]:
        print("case 2 np: ", method)
        x = np.copy(x_rand)
        x, _ = utils.np_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()

        print("case 2 custom: ", method, " , without acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
            1,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()

        print("case 2 custom: ", method, " , with acceleration")
        x = np.copy(x_rand)
        x, _ = utils.custom_nonlinear_CG(
            x,
            1e-8,
            0.5,
            0.8,
            utils.nonlinear_func_2,
            utils.grad(utils.nonlinear_func_2),
            method,
            2,
        )
        x = np.array(x)
        x_min = np.zeros(100)
        assert np.isclose(x_min, x).all()


if __name__ == "__main__":
    test_linear_cg()
    test_nonlinear_cg()
