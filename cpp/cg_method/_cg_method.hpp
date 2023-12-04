#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <vector>
#include <_matrix.hpp>
#include <algorithm>
#include <cmath>

namespace cg_method{

class linear_CG{
    public:
        linear_CG(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x) : A(A), b(b), x(x) {};
        linear_CG(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x, double epsilon) : A(A), b(b), x(x), epsilon(epsilon) {};
        linear_CG(Matrix::Naive_Matrix A, Matrix::Naive_Matrix b, Matrix::Naive_Matrix x, double epsilon, double epoch) : A(A), b(b), x(x), epsilon(epsilon), epoch(epoch) {};
        Matrix::Naive_Matrix solve();
        Matrix::Naive_Matrix const & get_A() const{ return A; }
        Matrix::Naive_Matrix       & set_A(){ return A; }
        Matrix::Naive_Matrix const & get_b() const{ return b; }
        Matrix::Naive_Matrix       & set_b(){ return b; }
        Matrix::Naive_Matrix const & get_x() const{ return x; }
        Matrix::Naive_Matrix       & set_x(){ return x; }
        double const & get_epsilon() const{ return epsilon; }
        double       & set_epsilon(){ return epsilon; }
        double const & get_epoch() const{ return epoch; }
        double       & set_epoch(){ return epoch; }

    private:
        Matrix::Naive_Matrix A;
        Matrix::Naive_Matrix b;
        Matrix::Naive_Matrix x;
        Matrix::Naive_Matrix res;
        Matrix::Naive_Matrix delta;
        Matrix::Naive_Matrix D;
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