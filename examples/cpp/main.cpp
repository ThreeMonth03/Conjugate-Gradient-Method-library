#include <cg/cg.hpp>

#include "checks.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace
{

void check_prepared_solver(const cg::PreparedSolver &solver)
{
    auto workspace = solver.create_workspace();
    cg::CgOptions options;
    options.rtol = 1e-12;
    options.maxiter = 10;

    for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
    {
        if (algorithm == cg::Algorithm::explicit_residual && solver.preconditioner() == cg::PreconditionerKind::jacobi)
        {
            continue;
        }
        options.algorithm = algorithm;
        for (const auto &b : example::right_hand_sides)
        {
            std::array<double, 2> x = {};
            const auto info = solver.solve_into(b, x, workspace, options);
            example::check_solution(b, x, info);
        }

        const auto &b = example::right_hand_sides.front();
        auto x = example::exact_solution(b);
        const auto converged = solver.solve_into(b, x, workspace, options);
        example::check_solution(b, x, converged);
        example::check_initial_counts(converged, cg::Status::converged);

        x.fill(0.0);
        auto limited = options;
        limited.maxiter = 0;
        const auto exhausted = solver.solve_into(b, x, workspace, limited);
        example::check_initial_counts(exhausted, cg::Status::max_iterations);
        example::check_zero_output(x);
    }
}

} // namespace

int main()
{
    try
    {
        const std::vector<double> values(example::matrix_values.begin(), example::matrix_values.end());
        const int num_threads = cg::CpuBackend::openmp_enabled() ? 2 : 1;
        for (const auto preconditioner : {cg::PreconditionerKind::none, cg::PreconditionerKind::jacobi})
        {
            const cg::PreparedSolver dense(cg::DenseMatrix(2, 2, values), num_threads, preconditioner);
            check_prepared_solver(dense);

            const cg::PreparedSolver csr(cg::CsrMatrix(2, 2, {0, 2, 4}, {0, 1, 0, 1}, values), num_threads,
                                         preconditioner);
            check_prepared_solver(csr);

            if (cg::CpuBackend::blas_enabled())
            {
                const cg::PreparedSolver blas(cg::DenseMatrix(2, 2, values), 1, preconditioner, cg::DenseKernel::blas);
                check_prepared_solver(blas);
            }
        }
        std::cout << "CPU consumer passed (OpenMP: " << cg::CpuBackend::openmp_enabled()
                  << ", BLAS: " << cg::CpuBackend::blas_enabled() << ")\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CPU consumer failed: " << error.what() << '\n';
        return 1;
    }
}
