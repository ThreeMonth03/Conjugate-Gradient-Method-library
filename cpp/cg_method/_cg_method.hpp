#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <vector>
#include <_matrix.hpp>
#include <algorithm>
#include <cmath>

namespace cg_method{

class linear_CG{
    public:
        linear_CG() = default;
        linear_CG(double epsilon) : epsilon(epsilon) {};
        linear_CG(double epsilon, double epoch) : epsilon(epsilon), epoch(epoch) {};
        Matrix::Naive_Matrix solve_by_Naive_Matrix(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x);
        double const & get_epsilon() const{ return epsilon; }
        double       & set_epsilon(){ return epsilon; }
        double const & get_epoch() const{ return epoch; }
        double       & set_epoch(){ return epoch; }

    private:
        double epsilon = 5e-7;
        double beta = 0;
        double chi = 0;
        double epoch = 10000000;
};

class nonlinear_CG{
    public:
        static Matrix::Naive_Matrix Fletcher_Reeves_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Naive_Matrix Polak_Ribiere_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Naive_Matrix Hager_Zhang_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
        static Matrix::Naive_Matrix Dai_Yuan_next_iteration(Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta);
};

}
#endif