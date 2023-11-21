#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <vector>
#include "../matrix/_matrix.hpp"

class linear_CG{
    public:
        linear_CG(Matrix A, Matrix b, Matrix x) : A(A), b(b), x(x) {};
        linear_CG(Matrix A, Matrix b, Matrix x, double epsilon) : A(A), b(b), x(x), epsilon(epsilon) {};
        linear_CG(Matrix A, Matrix b, Matrix x, double epsilon, double epoch) : A(A), b(b), x(x), epsilon(epsilon), epoch(epoch) {};
        Matrix solve();
        Matrix const & get_A() const{ return A; }
        Matrix       & set_A(){ return A; }
        Matrix const & get_b() const{ return b; }
        Matrix       & set_b(){ return b; }
        Matrix const & get_x() const{ return x; }
        Matrix       & set_x(){ return x; }
        double const & get_epsilon() const{ return epsilon; }
        double       & set_epsilon(){ return epsilon; }
        double const & get_epoch() const{ return epoch; }
        double       & set_epoch(){ return epoch; }

    private:
        Matrix A;
        Matrix b;
        Matrix x;
        Matrix res;
        Matrix delta;
        Matrix D;
        double epsilon = 5e-7;
        double beta = 0;
        double chi = 0;
        double epoch = 10000000;
};

#endif