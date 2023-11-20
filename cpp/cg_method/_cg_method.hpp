#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../matrix/_matrix.hpp"

class linear_CG{
    public:
        linear_CG(Matrix A, Matrix b, Matrix x,double epsilon, double epoch);
        void solve();
        Matrix get_x();
        Matrix get_res();
        Matrix get_delta();
        Matrix get_D();

    private:
        Matrix A;
        Matrix b;
        Matrix x;
        Matrix res;
        Matrix delta;
        Matrix D;
        double epsilon;
        double beta;
        double chi;
        double epoch = 100000;
}

#endif