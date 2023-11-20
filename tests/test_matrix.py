import pytest
import math
import _matrix
import time

def test_matrix_oop():
    mat1 = _matrix.Matrix(2, 2)
    mat4 = _matrix.Matrix(1000, 1000)
    assert(mat1.nrow == mat1.ncol == 2)
    assert(mat4.nrow == mat4.ncol == 1000)
    arr = [1, 0, 0, 1]
    mat3 = _matrix.Matrix(2, 2, arr)
    for i in range(mat3.nrow):
        for j in range(mat3.ncol):
            assert(arr[i*mat3.nrow+j] == mat3[i, j])
    mat4 = mat1 + mat3
    assert(mat4 == mat3)

def test_mul():
    mat1 = _matrix.Matrix(1000, 1000)
    mat2 = _matrix.Matrix(1000, 1000)
    for i in range(1000):
        for j in range(1000):
            mat1[i,j] = 1.0
            mat2[i,j] = 2.0
    mat_naive = _matrix.multiply_naive(mat1, mat2)
    mat_tile = _matrix.multiply_tile(mat1, mat2, 32)

    for i in range(1000):
        for j in range(1000):
            assert(mat_naive[i,j] == 2000)   
            assert(mat_tile[i,j] == 2000) 
