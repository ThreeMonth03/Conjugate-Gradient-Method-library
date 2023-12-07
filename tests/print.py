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
from _cgpy.Matrix import Naive_Matrix
from _cgpy.Matrix import Accelerated_Matrix
import time
import numpy as np

def test_naive_matrix_func():
    np3 = np.random.rand(10, 10)
    mat3 = Naive_Matrix(np3)
    assert(mat3.nrow == mat3.ncol == np3.shape[0] == np3.shape[1])
    for i in range(mat3.nrow):
        for j in range(mat3.ncol):
            assert(mat3[i, j] == np3[i, j])
            print(mat3[i, j], " ",np3[i, j])

if __name__ == "__main__":
    test_naive_matrix_func()