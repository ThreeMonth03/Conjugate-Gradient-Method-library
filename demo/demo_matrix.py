import pytest
import math
import _matrix
import time

def performance_comp(matrix_size = 1024, epoch = 10):
    size = matrix_size
    total_epoch = epoch
    mat1 = _matrix.Matrix(size, size)
    mat2 = _matrix.Matrix(size, size)
    for i in range(size):
        for j in range(size):
            mat1[i,j] = 1.0
            mat2[i,j] = 2.0
    
    naive_total_time = 0
    tile_total_time = 0

    for idx, s in enumerate(["naive_multiplex", "tile_multiplex"]):
        for i in range(total_epoch):
            total = 0
            if(idx == 0):
                start = time.process_time()
                mattemp = _matrix.multiply_naive(mat1, mat2)
                end = time.process_time()
                total = end - start
                naive_total_time += total

            if(idx == 1):
                start = time.process_time()
                mattemp = _matrix.multiply_tile(mat1, mat2, 64)
                end = time.process_time()
                total = end - start
                tile_total_time += total

            print(s," for epoch ",i, " : ", total,"s")
    naive_total_time /= total_epoch
    tile_total_time /= total_epoch

    print("Average time for naive_multiplex with matrix size", size, "*",size,": ", naive_total_time,"s")                
    print("Average time for tile_multiplex with matrix size", size, "*",size,": ", tile_total_time,"s") 
    print("Tile is ",naive_total_time/tile_total_time, " times faster than naive.")

if __name__ == '__main__':
    performance_comp(1024, 1)