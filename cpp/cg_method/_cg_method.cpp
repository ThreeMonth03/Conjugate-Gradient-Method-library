#include <_cg_method.hpp>
#include <algorithm>
#include <cg/solver.hpp>
#include <cmath>
#include <stdexcept>
#include <string>

namespace cg_method
{

namespace
{
int max_threads()
{
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

template <class MatrixType>
void solve_legacy(const MatrixType &A, const MatrixType &b, MatrixType &x, double epsilon,
                  int epoch, int threads, bool parallel)
{
    if (epoch < 0)
        throw std::invalid_argument("Maximum iteration count must be nonnegative");
    if (threads < 1)
        throw std::invalid_argument("Thread count must be positive");
    if (!std::isfinite(epsilon) || epsilon < 0.0)
        throw std::invalid_argument("Residual tolerance must be finite and nonnegative");
    if (A.nrow() == 0 || A.nrow() != A.ncol() || b.ncol() != 1 || x.ncol() != 1 ||
        b.nrow() != A.nrow() || x.nrow() != A.nrow())
        throw std::invalid_argument(
            "CG requires a nonempty square matrix and matching column vectors");

    const int effective_threads = parallel && cg::CpuBackend::openmp_enabled() ? threads : 1;
    const cg::CpuBackend backend(effective_threads);
    cg::Workspace workspace(A.nrow());
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    options.rtol = 0.0;
    options.atol = epsilon;
    options.maxiter = static_cast<std::size_t>(epoch);
    const auto info = cg::solve_into({A.data(), A.nrow(), A.ncol()}, {b.data(), b.size()},
                                     {x.data(), x.size()}, workspace, backend, options);
    if (info.status == cg::Status::non_finite || info.status == cg::Status::non_positive_curvature)
        throw std::runtime_error(std::string("CG numerical failure: ") +
                                 cg::status_name(info.status));
}
} // namespace

Matrix::Naive_Matrix linear_CG::solve_by_Naive_Matrix(const Matrix::Naive_Matrix &A,
                                                      const Matrix::Naive_Matrix &b,
                                                      Matrix::Naive_Matrix x)
{
    solve_legacy(A, b, x, epsilon, epoch, number_of_threads, false);
    return x;
}

Matrix::Accelerated_Matrix
linear_CG::solve_by_Accelerated_Matrix(const Matrix::Accelerated_Matrix &A,
                                       const Matrix::Accelerated_Matrix &b,
                                       Matrix::Accelerated_Matrix x)
{
    solve_legacy(A, b, x, epsilon, epoch, number_of_threads, true);
    return x;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Fletcher_Reeves_next_iteration(
    Matrix::Naive_Matrix cur_Df, Matrix::Naive_Matrix next_Df, Matrix::Naive_Matrix delta)
{
    /*
    Functionality: calculate the next iteration of delta by Fletcher Reeves method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current
    iteration next_Df: Naive_Matrix, the gradient of the objective function at
    next iteration delta: Naive_Matrix, the direction of the next iteration
    */
    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Hager_Zhang_next_iteration(Matrix::Naive_Matrix cur_Df,
                                                                    Matrix::Naive_Matrix next_Df,
                                                                    Matrix::Naive_Matrix delta)
{
    /*
    Functionality: calculate the next iteration of delta by Hager Zhang method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current
    iteration next_Df: Naive_Matrix, the gradient of the objective function at
    next iteration delta: Naive_Matrix, the direction of the next iteration
    */
    Matrix::Naive_Matrix Q = next_Df - cur_Df;
    Matrix::Naive_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix::Naive_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Naive_Matrix nonlinear_CG::Naive_Dai_Yuan_next_iteration(Matrix::Naive_Matrix cur_Df,
                                                                 Matrix::Naive_Matrix next_Df,
                                                                 Matrix::Naive_Matrix delta)
{
    /*
    Functionality: calculate the next iteration of delta by Dai Yuan method
    Parameter:
        cur_Df: Naive_Matrix, the gradient of the objective function at current
    iteration next_Df: Naive_Matrix, the gradient of the objective function at
    next iteration delta: Naive_Matrix, the direction of the next iteration
    */
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Fletcher_Reeves_next_iteration(
    Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df,
    Matrix::Accelerated_Matrix delta, int number_of_threads)
{
    /*
    Functionality: calculate the next iteration of delta by Fletcher Reeves method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at
    current iteration next_Df: Accelerated_Matrix, the gradient of the objective
    function at next iteration delta: Accelerated_Matrix, the direction of the
    next iteration
    */
    cur_Df.set_number_of_threads() = number_of_threads == -1 ? max_threads() : number_of_threads;

    double chi = std::pow(next_Df.norm() / cur_Df.norm(), 2);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Hager_Zhang_next_iteration(
    Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df,
    Matrix::Accelerated_Matrix delta, int number_of_threads)
{
    /*
    Functionality: calculate the next iteration of delta by Hager Zhang method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at
    current iteration next_Df: Accelerated_Matrix, the gradient of the objective
    function at next iteration delta: Accelerated_Matrix, the direction of the
    next iteration
    */

    cur_Df.set_number_of_threads() = number_of_threads == -1 ? max_threads() : number_of_threads;

    Matrix::Accelerated_Matrix Q = next_Df - cur_Df;
    Matrix::Accelerated_Matrix M = Q - delta * (std::pow(Q.norm(), 2) * 2 / (delta * Q)(0, 0));
    Matrix::Accelerated_Matrix N = next_Df * (1 / (delta * Q)(0, 0));
    double chi = (M * N)(0, 0);
    delta = -next_Df + delta * chi;
    return delta;
}

Matrix::Accelerated_Matrix nonlinear_CG::Accelerated_Dai_Yuan_next_iteration(
    Matrix::Accelerated_Matrix cur_Df, Matrix::Accelerated_Matrix next_Df,
    Matrix::Accelerated_Matrix delta, int number_of_threads)
{
    /*
    Functionality: calculate the next iteration of delta by Dai Yuan method
    Parameter:
        cur_Df: Accelerated_Matrix, the gradient of the objective function at
    current iteration next_Df: Accelerated_Matrix, the gradient of the objective
    function at next iteration delta: Accelerated_Matrix, the direction of the
    next iteration
    */

    cur_Df.set_number_of_threads() = number_of_threads == -1 ? max_threads() : number_of_threads;
    double chi = std::pow(next_Df.norm(), 2) / (delta * (next_Df - cur_Df))(0, 0);
    delta = -next_Df + delta * chi;
    return delta;
}

} // namespace cg_method
