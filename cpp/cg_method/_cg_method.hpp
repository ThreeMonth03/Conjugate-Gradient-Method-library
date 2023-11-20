#ifndef __CG_METHOD_HPP__
#define __CG_METHOD_HPP__

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../matrix/_matrix.hpp"

class linear_CG{
    public:
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
}

#endif