import pytest
import math
import _matrix
import time
import numpy as np

def test_matrix_():
    mat1 = _matrix.Matrix(2, 2)
    assert(mat1.nrow == mat1.ncol == 2)
    arr = [1, 0, 0, 1]
    mat2 = _matrix.Matrix(2, 2, arr)
    for i in range(mat2.nrow):
        for j in range(mat2.ncol):
            assert(arr[i*mat2.nrow+j] == mat2[i, j])

    mat3 = _matrix.Matrix(1000, 1000)
    assert(mat3.nrow == mat3.ncol == 1000)
    for i in range(mat3.nrow):
        for j in range(mat3.ncol):
            assert(mat3[i, j] == 0)
    
    np_mat1 = np.random.rand(1000, 1000)
    np_mat1_flatten = np_mat1.flatten()
    mat4 = _matrix.Matrix(1000, 1000, np_mat1_flatten)
    np_mat4 = np.array(mat4.tolist())
    assert(np_mat4.shape == np_mat1_flatten.shape)
    assert((np_mat4 == np_mat1_flatten).all)

def test_matrix_add():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 + np_mat2
    np_flatten_mat3 = np_mat3.flatten()

    mat1 = _matrix.Matrix(np_mat1.shape[0], np_mat1.shape[1], np_mat1.flatten())
    mat2 = _matrix.Matrix(np_mat2.shape[0], np_mat2.shape[1], np_mat2.flatten())
    mat_add = mat1 + mat2
    np_mat_add = np.array(mat_add.tolist())

    assert(np_mat_add.shape == np_flatten_mat3.shape)
    assert(np.isclose(np_mat_add, np_flatten_mat3).all)

def test_matrix_sub():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1 - np_mat2
    np_flatten_mat3 = np_mat3.flatten()

    mat1 = _matrix.Matrix(np_mat1.shape[0], np_mat1.shape[1], np_mat1.flatten())
    mat2 = _matrix.Matrix(np_mat2.shape[0], np_mat2.shape[1], np_mat2.flatten())
    mat_add = mat1 - mat2
    np_mat_add = np.array(mat_add.tolist())

    assert(np_mat_add.shape == np_flatten_mat3.shape)
    assert(np.isclose(np_mat_add, np_flatten_mat3).all)

def test_matrix_neg():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = -np_mat1
    np_flatten_mat2 = np_mat2.flatten()

    mat1 = _matrix.Matrix(np_mat1.shape[0], np_mat1.shape[1], np_mat1.flatten())
    mat_neg = -mat1
    np_mat_neg = np.array(mat_neg.tolist())

    assert(np_mat_neg.shape == np_flatten_mat2.shape)
    assert(np.isclose(np_mat_neg, np_flatten_mat2).all)


def test_matmul():
    np_mat1 = np.random.rand(1000, 1000)
    np_mat2 = np.random.rand(1000, 1000)
    np_mat3 = np_mat1.dot(np_mat2)
    np_flatten_mat3 = np_mat3.flatten()

    mat1 = _matrix.Matrix(np_mat1.shape[0], np_mat1.shape[1], np_mat1.flatten())
    mat2 = _matrix.Matrix(np_mat2.shape[0], np_mat2.shape[1], np_mat2.flatten())

    mat_dot = mat1 @ mat2
    mat_naive = _matrix.multiply_naive(mat1, mat2)
    mat_tile = _matrix.multiply_tile(mat1, mat2, 32)
   
    np_mat_dot = np.array(mat_dot.tolist())
    np_mat_naive = np.array(mat_naive.tolist())
    np_mat_tile = np.array(mat_tile.tolist())
    
    assert(np_mat_dot.shape == np_flatten_mat3.shape)
    assert(np_mat_naive.shape == np_flatten_mat3.shape)
    assert(np_mat_tile.shape == np_flatten_mat3.shape)

    assert(np.isclose(np_mat_dot, np_flatten_mat3).all)
    assert(np.isclose(np_mat_naive, np_flatten_mat3).all)
    assert(np.isclose(np_mat_tile, np_flatten_mat3).all) 

def test_mul():
    np_mat1 = np.random.rand(1000, 1000)
    scalar = np.random.rand()
    np_mat2 = np_mat1 * scalar
    np_flatten_mat2 = np_mat2.flatten()

    mat1 = _matrix.Matrix(np_mat1.shape[0], np_mat1.shape[1], np_mat1.flatten())
    mat_mul = mat1 * scalar
    np_mat_mul = np.array(mat_mul.tolist())
    assert(np_mat_mul.shape == np_flatten_mat2.shape)
    assert(np.isclose(np_mat_mul, np_flatten_mat2).all)
