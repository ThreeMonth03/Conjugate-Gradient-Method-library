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

## Test Naive Matrix 

def test_naive_matrix_func():
    mat1 = Naive_Matrix(2, 2)
    assert(mat1.nrow == mat1.ncol == 2)
    arr = [1, 0, 0, 1]
    mat2 = Naive_Matrix(2, 2, arr)
    for i in range(mat2.nrow):
        for j in range(mat2.ncol):
            assert(arr[i*mat2.nrow+j] == mat2[i, j])

    mat3 = Naive_Matrix(1000, 1000)
    assert(mat3.nrow == mat3.ncol == 1000)
    for i in range(mat3.nrow):
        for j in range(mat3.ncol):
            assert(mat3[i, j] == 0)
    
    np_mat1 = np.random.rand(1000, 1000)
    np_mat1_flatten = np_mat1.flatten()
    mat4 = Naive_Matrix(1000, 1000, np_mat1_flatten)
    np_mat4 = np.array(mat4.tolist())
    assert(np_mat4.shape == np_mat1_flatten.shape)
    assert((np_mat4 == np_mat1_flatten).all)

    mat5 = Naive_Matrix(np_mat1)
    np_mat5 = np.array(mat5.tolist2d())
    assert(np_mat5.shape == np_mat1.shape)
    assert((np_mat5 == np_mat1).all)

def test_naive_matrix_add():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 + np_mat2
    
    mat1 = Naive_Matrix(np_mat1)
    mat2 = Naive_Matrix(np_mat2)
    mat_add = mat1 + mat2
    np_mat_add = np.array(mat_add.tolist2d())

    assert(np_mat_add.shape == np_mat3.shape)
    assert(np.isclose(np_mat_add, np_mat3).all)

def test_naive_matrix_sub():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 - np_mat2

    mat1 = Naive_Matrix(np_mat1)
    mat2 = Naive_Matrix(np_mat2)
    mat_add = mat1 - mat2
    np_mat_add = np.array(mat_add.tolist2d())

    assert(np_mat_add.shape == np_mat3.shape)
    assert(np.isclose(np_mat_add, np_mat3).all)

def test_naive_matrix_neg():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = -np_mat1

    mat1 = Naive_Matrix(np_mat1)
    mat_neg = -mat1
    np_mat_neg = np.array(mat_neg.tolist2d())

    assert(np_mat_neg.shape == np_mat2.shape)
    assert(np.isclose(np_mat_neg, np_mat2).all)


def test_naive_matrix_matmul():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1.dot(np_mat2)

    mat1 = Naive_Matrix(np_mat1)
    mat2 = Naive_Matrix(np_mat2)

    mat_dot = mat1 @ mat2
    np_mat_dot = np.array(mat_dot.tolist2d())
    assert(np_mat_dot.shape == np_mat3.shape)
    assert(np.isclose(np_mat_dot, np_mat3).all)

def test_naive_matrix_matmul_broadcast():
    ## 1d array dot 1d array
    np_mat1 = np.random.rand(1,1000)
    np_mat2 = np.random.rand(1000, 1)
    np_mat3 = np_mat1.dot(np_mat2)

    assert(np_mat3.shape == (1,1))
    mat1 = Naive_Matrix(np_mat1.T)
    mat2 = Naive_Matrix(np_mat2)

    mat_dot = mat1 @ mat2
    assert(np.isclose(mat_dot.tolist2d(), np_mat3))

    #scalar dot 2d array
    np_mat1 = np.random.rand()
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 * np_mat2
    np_mat1 = np.array([[np_mat1]])
    assert(np_mat3.shape == (1000, 1000))

    mat1 = Naive_Matrix(np_mat1)
    mat2 = Naive_Matrix(np_mat2)
    mat_dot = mat1 @ mat2
    np_mat_dot = np.array(mat_dot.tolist2d())
    assert(np_mat_dot.shape == np_mat3.shape)
    assert(np.isclose(np_mat_dot, np_mat3).all)

    #2d array dot scalar
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand()
    np_mat3 = np_mat1 * np_mat2
    np_mat2 = np.array([[np_mat2]])

    mat1 = Naive_Matrix(np_mat1)
    mat2 = Naive_Matrix(np_mat2)
    mat_mul = mat1 @ mat2
    np_mat_mul = np.array(mat_mul.tolist2d())
    assert(np_mat_mul.shape == np_mat3.shape)
    assert(np.isclose(np_mat_mul, np_mat3).all)


def test_naive_matrix_mul():
    np_mat1 = np.random.rand(1000, 1000)
    scalar = np.random.rand()
    np_mat2 = np_mat1 * scalar

    mat1 = Naive_Matrix(np_mat1)
    mat_mul = mat1 * scalar
    np_mat_mul = np.array(mat_mul.tolist2d())
    assert(np_mat_mul.shape == np_mat2.shape)
    assert(np.isclose(np_mat_mul, np_mat2).all)

def test_naive_matrix_norm():
    np_mat1 = np.random.rand(1000, 1000)
    np_norm = np.linalg.norm(np_mat1)
    mat1 = Naive_Matrix(np_mat1)
    mat1_norm = mat1.norm()
    assert(np.isclose(np_norm, mat1_norm))

## Test Accelerated Matrix

def test_accelerated_matrix_func():
    mat1 = Accelerated_Matrix(2, 2)
    assert(mat1.nrow == mat1.ncol == 2)
    arr = [1, 0, 0, 1]
    mat2 = Accelerated_Matrix(2, 2, arr)
    for i in range(mat2.nrow):
        for j in range(mat2.ncol):
            assert(arr[i*mat2.nrow+j] == mat2[i, j])

    mat3 = Accelerated_Matrix(1000, 1000)
    assert(mat3.nrow == mat3.ncol == 1000)
    for i in range(mat3.nrow):
        for j in range(mat3.ncol):
            assert(mat3[i, j] == 0)
    
    np_mat1 = np.random.rand(1000, 1000)
    np_mat1_flatten = np_mat1.flatten()
    mat4 = Accelerated_Matrix(1000, 1000, np_mat1_flatten)
    np_mat4 = np.array(mat4.tolist())
    assert(np_mat4.shape == np_mat1_flatten.shape)
    assert((np_mat4 == np_mat1_flatten).all)

    mat5 = Accelerated_Matrix(np_mat1)
    np_mat5 = np.array(mat5.tolist2d())
    assert(np_mat5.shape == np_mat1.shape)
    assert((np_mat5 == np_mat1).all)

def test_accelerated_matrix_add():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 + np_mat2

    mat1 = Accelerated_Matrix(np_mat1)
    mat2 = Accelerated_Matrix(np_mat2)
    mat_add = mat1 + mat2
    np_mat_add = np.array(mat_add.tolist2d())

    assert(np_mat_add.shape == np_mat3.shape)
    assert(np.isclose(np_mat_add, np_mat3).all)

def test_accelerated_matrix_sub():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 - np_mat2

    mat1 = Accelerated_Matrix(np_mat1)
    mat2 = Accelerated_Matrix(np_mat2)
    mat_add = mat1 - mat2
    np_mat_add = np.array(mat_add.tolist2d())

    assert(np_mat_add.shape == np_mat3.shape)
    assert(np.isclose(np_mat_add, np_mat3).all)

def test_accelerated_matrix_neg():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = -np_mat1

    mat1 = Accelerated_Matrix(np_mat1)
    mat_neg = -mat1
    np_mat_neg = np.array(mat_neg.tolist2d())

    assert(np_mat_neg.shape == np_mat2.shape)
    assert(np.isclose(np_mat_neg, np_mat2).all)


def test_accelerated_matrix_matmul():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1.dot(np_mat2)

    mat1 = Accelerated_Matrix(np_mat1)
    mat2 = Accelerated_Matrix(np_mat2)

    mat_dot = mat1 @ mat2
    np_mat_dot = np.array(mat_dot.tolist2d())
    assert(np_mat_dot.shape == np_mat3.shape)
    assert(np.isclose(np_mat_dot, np_mat3).all)

def test_accelerated_matrix_matmul_broadcast():
    ## 1d array dot 1d array
    np_mat1 = np.random.rand(1,1000)
    np_mat2 = np.random.rand(1000, 1)
    np_mat3 = np_mat1.dot(np_mat2)

    assert(np_mat3.shape == (1,1))
    mat1 = Accelerated_Matrix(np_mat1.T)
    mat2 = Accelerated_Matrix(np_mat2)

    mat_dot = mat1 @ mat2
    assert(np.isclose(mat_dot.tolist2d(), np_mat3))

    #scalar dot 2d array
    np_mat1 = np.random.rand()
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 * np_mat2
    np_mat1 = np.array([[np_mat1]])
    assert(np_mat3.shape == (1000, 1000))

    mat1 = Accelerated_Matrix(np_mat1)
    mat2 = Accelerated_Matrix(np_mat2)
    mat_dot = mat1 @ mat2
    np_mat_dot = np.array(mat_dot.tolist2d())
    assert(np_mat_dot.shape == np_mat3.shape)
    assert(np.isclose(np_mat_dot, np_mat3).all)

    #2d array dot scalar
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand()
    np_mat3 = np_mat1 * np_mat2
    np_mat2 = np.array([[np_mat2]])

    mat1 = Accelerated_Matrix(np_mat1)
    mat2 = Accelerated_Matrix(np_mat2)
    mat_mul = mat1 @ mat2
    np_mat_mul = np.array(mat_mul.tolist2d())
    assert(np_mat_mul.shape == np_mat3.shape)
    assert(np.isclose(np_mat_mul, np_mat3).all)


def test_accelerated_matrix_mul():
    np_mat1 = np.random.rand(1000, 1000)
    scalar = np.random.rand()
    np_mat2 = np_mat1 * scalar

    mat1 = Accelerated_Matrix(np_mat1)
    mat_mul = mat1 * scalar
    np_mat_mul = np.array(mat_mul.tolist2d())
    assert(np_mat_mul.shape == np_mat2.shape)
    assert(np.isclose(np_mat_mul, np_mat2).all)

def test_accelerated_matrix_norm():
    np_mat1 = np.random.rand(1000, 1000)
    np_norm = np.linalg.norm(np_mat1)
    mat1 = Accelerated_Matrix(np_mat1)
    mat1_norm = mat1.norm()
    assert(np.isclose(np_norm, mat1_norm))