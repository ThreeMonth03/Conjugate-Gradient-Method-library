#include "_cg_method.hpp"

Matrix linear_CG::solve(){
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