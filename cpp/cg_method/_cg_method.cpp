#include "_cg_method.hpp"
#include <algorithm>
#include <cmath>

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

Matrix nonlinear_CG::Fletcher_Reeves_next_iteration(Matrix cur_Df, Matrix next_Df, Matrix delta){
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix nonlinear_CG::Polak_Ribiere_next_iteration(Matrix cur_Df, Matrix next_Df, Matrix delta){
    double chi = ((next_Df - cur_Df) * next_Df)(0, 0) / std::pow(cur_Df.norm(), 2);
    chi = std::max(.0d, chi);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix nonlinear_CG::Hager_Zhang_next_iteration(Matrix cur_Df, Matrix next_Df, Matrix delta){
    Matrix Q = next_Df - cur_Df;
    Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = - next_Df + delta * chi;
    return delta;    
}

Matrix nonlinear_CG::Dai_Yuan_next_iteration(Matrix cur_Df, Matrix next_Df, Matrix delta){
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = - next_Df + delta * chi;
    return delta;
}