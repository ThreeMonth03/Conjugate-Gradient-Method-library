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

def compare_liear_cg(epoch,  num_of_threads = 16):
    print("Compare linear CG")
    np.random.seed(0)
    a = np.random.rand(500, 500)
    a = utils.generate_pos_def(500)
    a = utils.generate_symmetric(500)
    while(np.linalg.cond(a) < 10000):
        a = np.random.rand(500, 500)
        a = utils.generate_pos_def(500)
        a = utils.generate_symmetric(500)
    print("find a good matrix")
    b = np.random.rand(500)
    x = np.random.rand(500)

    sum = 0.0
    for i in range(epoch):
        total = 0.0
        start = time.time()
        x = utils.np_linear_CG(x, a, b, 5e-7)
        end = time.time()
        total += end - start
        sum += total
        #print("Numpy for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Numpy average time: ", np_total_avg)
    
    sum = 0.0
    for i in range(epoch):
        total = 0.0
        start = time.time()
        np_naive_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, num_threads = 1)
        end = time.time()
        total += end - start
        sum += total
        #print("Naive for epoch ", i, " takes ", total, " seconds")
    np_total_avg = sum / epoch
    print("Naive average time: ", np_total_avg)

    for i in range(2, num_of_threads):
        sum = 0.0
        for j in range(epoch):
            total = 0.0
            start = time.time()
            np_acc_mat_x_min = utils.custom_linear_CG(x = x, a = a, b = b,  epsilon = 5e-7, epoch=10000, num_threads = j)
            end = time.time()
            total += end - start
            sum += total
            #print("Accelerated for epoch ", j, "and ", i," threads takes ", total, " seconds")
        np_total_avg = sum / epoch
        print("Accelerated average time for ", i, " threads: ", np_total_avg)

if __name__ == '__main__':
    compare_liear_cg(10, 16)