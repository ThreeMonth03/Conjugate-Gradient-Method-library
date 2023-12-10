#include <_cg_method.hpp>
#include <algorithm>
#include <cmath>

namespace cg_method{

Matrix::Naive_Matrix linear_CG::solve_by_Naive_Matrix(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x){
    Matrix::Naive_Matrix res;
    Matrix::Naive_Matrix delta;
    Matrix::Naive_Matrix D;
    res = A * x - b; //1d array
    delta = -res; //1d array
    for (int i = 0; i < epoch; ++i)
    {
        if (res.norm() < epsilon)
        {
            break;
        }
        D = A * delta; //1d array
        beta = -(res * delta)(0, 0) / (delta * D)(0, 0); //scalar
        x = x + (delta * beta); //1d array
        res = A * x - b; //1d array
        chi = (res * D)(0, 0) / (delta * D)(0, 0); //scalar
        delta = delta * chi -  res; //1d array
    }
    return x;
}

Matrix::Accelerated_Matrix linear_CG::solve_by_Accelerated_Matrix(Matrix::Accelerated_Matrix A, Matrix::Accelerated_Matrix b, Matrix::Accelerated_Matrix x){
    Matrix::Accelerated_Matrix res;
    Matrix::Accelerated_Matrix delta;
    Matrix::Accelerated_Matrix D;

    A.set_number_of_threads() = number_of_threads;
    b.set_number_of_threads() = number_of_threads;
    x.set_number_of_threads() = number_of_threads;
    res.set_number_of_threads() = number_of_threads;
    delta.set_number_of_threads() = number_of_threads;
    D.set_number_of_threads() = number_of_threads;

    res = A * x - b; //1d array
    delta = -res; //1d array
    for (int i = 0; i < epoch; ++i)
    {
        if (res.norm() < epsilon)
        {
            break;
        }
        D = A * delta; //1d array
        beta = -(res * delta)(0, 0) / (delta * D)(0, 0); //scalar
        x = x + (delta * beta); //1d array
        res = A * x - b; //1d array
        chi = (res * D)(0, 0) / (delta * D)(0, 0); //scalar
        delta = delta * chi -  res; //1d array
    }
    return x;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Fletcher_Reeves_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Hager_Zhang_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    Matrix::Naive_Matrix Q = next_Df - cur_Df;
    Matrix::Naive_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix::Naive_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Dai_Yuan_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta){
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = - next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Fletcher_Reeves_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    next_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Hager_Zhang_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    next_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    
    Matrix::Accelerated_Matrix Q = next_Df - cur_Df;
    Q.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    Matrix::Accelerated_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    M.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    Matrix::Accelerated_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    N.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    double chi = (M * N)(0, 0);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Dai_Yuan_next_iteration(Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df, Matrix::Accelerated_Matrix delta, int number_of_threads){
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    next_Df.set_number_of_threads() = number_of_threads == -1 ? omp_get_max_threads() : number_of_threads;
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = - next_Df + delta * chi;
    return delta;
}

}