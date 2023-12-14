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

def compare_matrix_2D_dot_2D(matrix_size = 1024, epoch = 5):
    print("Compare matrix multiplication 2D @ 2D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, size)
    np_mat2 = np.random.rand(size, size)
    
    
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1 @ np_mat2
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)
    Naive_mat2 = Naive_Matrix(np_mat2)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 @ Naive_mat2
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)
    Accelerated_mat2 = Accelerated_Matrix(np_mat2)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        Accelerated_mat2.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 @ Accelerated_mat2
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_1D_dot_1D(matrix_size = 1000000, epoch = 5):
    print("Compare matrix multiplication 1D @ 1D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, 1)
    np_mat2 = np.random.rand(size, 1)
    
    
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1.T @ np_mat2
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)
    Naive_mat2 = Naive_Matrix(np_mat2)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 @ Naive_mat2
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)
    Accelerated_mat2 = Accelerated_Matrix(np_mat2)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        Accelerated_mat2.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 @ Accelerated_mat2
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_2D_dot_0D(matrix_size = 1024, epoch = 5):
    print("Compare matrix multiplication 2D @ 0D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, size)
    np_mat2 = np.random.rand(1, 1)
    a = 3
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1 * np_mat2
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)
    Naive_mat2 = Naive_Matrix(np_mat2)
    
    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 @ Naive_mat2
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)
    Accelerated_mat2 = Accelerated_Matrix(np_mat2)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 @ Accelerated_mat2
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_2D_dot_scalar(matrix_size = 1024, epoch = 5):
    print("Compare matrix multiplication 2D * scalar")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, size) 
    a = 3
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1 * a
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 * a
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 * a
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_norm(matrix_size = 1000000, epoch = 5):
    print("Compare matrix norm")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, 1)
    
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np.linalg.norm(np_mat1)
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1.norm()
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1.norm()
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_1D_add_1D(matrix_size = 1000000, epoch = 5):
    print("Compare matrix addition 1D + 1D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, 1)
    np_mat2 = np.random.rand(size, 1)
    
    
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1 + np_mat2
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)
    Naive_mat2 = Naive_Matrix(np_mat2)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 + Naive_mat2
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)
    Accelerated_mat2 = Accelerated_Matrix(np_mat2)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        Accelerated_mat2.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 + Accelerated_mat2
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_1D_sub_1D(matrix_size = 1000000, epoch = 5):
    print("Compare matrix subtraction 1D - 1D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, 1)
    np_mat2 = np.random.rand(size, 1)
    
    
    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = np_mat1 - np_mat2
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)
    Naive_mat2 = Naive_Matrix(np_mat2)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = Naive_mat1 - Naive_mat2
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)
    Accelerated_mat2 = Accelerated_Matrix(np_mat2)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        Accelerated_mat2.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = Accelerated_mat1 - Accelerated_mat2
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

def compare_matrix_neg_1D(matrix_size = 1000000, epoch = 5):
    print("Compare matrix negative 1D")
    np.random.seed(0)
    size = matrix_size
    total_epoch = epoch
    ### Compare Numpy ###
    np_mat1 = np.random.rand(size, 1)
    
    
    sum = 0
    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        np_mattemp = -np_mat1
        end = time.process_time()
        total = end - start
        #print("Numpy for epoch ",i, " : ", total,"s")
        sum += total
    np_total_time = sum/total_epoch
    print("Average time for numpy with matrix size", size, "*",size,": ", np_total_time,"s")

    ### Compare Naive Matrix ###
    Naive_mat1 = Naive_Matrix(np_mat1)

    sum = 0

    for i in range(total_epoch):
        total = 0
        start = time.process_time()
        mattemp = -Naive_mat1
        end = time.process_time()
        total = end - start
        #print("Naive for epoch ",i, " : ", total,"s")
        sum += total
    naive_total_time = sum/total_epoch
    print("Average time for naive matrix with matrix size", size, "*",size,": ", naive_total_time,"s")
    
    ### Compare Accelerated Matrix ###
    Accelerated_mat1 = Accelerated_Matrix(np_mat1)


    for j in range(2,9):
        Accelerated_mat1.set_number_of_threads(j)
        print("Number of threads: ", j)
        sum = 0
        for i in range(total_epoch):
            total = 0
            start = time.process_time()
            mattemp = -Accelerated_mat1
            end = time.process_time()
            total = end - start
            #print("Accelerated for epoch ",i, " : ", total,"s")
            sum += total
        accelerated_total_time = sum/total_epoch
        print("Average time for accelerated matrix with matrix size", size, "*",size,": ", accelerated_total_time,"s")
    print("_____________________________")

if __name__ == '__main__':
    compare_matrix_2D_dot_2D(256, 100)
    #compare_matrix_1D_dot_1D(1000000,100)
    compare_matrix_2D_dot_0D(1024, 100)
    compare_matrix_2D_dot_scalar(1024, 100)
    #compare_matrix_norm(1000000, 100)
    #compare_matrix_1D_add_1D(1000000, 100)
    #compare_matrix_1D_sub_1D(1000000, 100)
    #compare_matrix_neg_1D(1000000, 100)