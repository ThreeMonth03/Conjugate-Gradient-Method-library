#include <cg/cg.hpp>
#include <cg/cuda.hpp>

#include "checks.hpp"

#include <array>
#include <exception>
#include <iostream>
#include <vector>

namespace
{

void check_prepared_solver(const cg::cuda::PreparedSolver &solver, const cg::cuda::Context &context)
{
    for (const auto mode : {cg::cuda::ExecutionMode::host, cg::cuda::ExecutionMode::device})
    {
        auto workspace = solver.create_workspace(context.stream(), mode);
        cg::cuda::Vector b(2, context);
        cg::cuda::Vector x(2, context);
        cg::CgOptions options;
        options.rtol = 1e-12;
        options.maxiter = 10;

        for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
        {
            if (algorithm == cg::Algorithm::explicit_residual &&
                solver.preconditioner() == cg::PreconditionerKind::jacobi)
            {
                continue;
            }
            options.algorithm = algorithm;
            for (const auto &rhs : example::right_hand_sides)
            {
                b.upload(rhs, context);
                x.fill(0.0, context);
                const auto info = solver.solve_into(b.view(), x.view(), workspace, options);
                std::array<double, 2> output;
                x.download(output, context);
                example::check_solution(rhs, output, info);
            }

            const auto &rhs = example::right_hand_sides.front();
            const auto exact = example::exact_solution(rhs);
            b.upload(rhs, context);
            x.upload(exact, context);
            const auto converged = solver.solve_into(b.view(), x.view(), workspace, options);
            std::array<double, 2> output;
            x.download(output, context);
            example::check_solution(rhs, output, converged);
            example::check_initial_counts(converged, cg::Status::converged);

            x.fill(0.0, context);
            auto limited = options;
            limited.maxiter = 0;
            const auto exhausted = solver.solve_into(b.view(), x.view(), workspace, limited);
            x.download(output, context);
            example::check_initial_counts(exhausted, cg::Status::max_iterations);
            example::check_zero_output(output);
        }
    }
}

} // namespace

int main()
{
    if (!cg::cuda::available())
    {
        std::cout << "CUDA consumer skipped: no CUDA device is available\n";
        return 77;
    }
    try
    {
        const cg::cuda::Context context;
        const std::vector<double> values(example::matrix_values.begin(), example::matrix_values.end());
        const cg::DenseMatrix host_dense(2, 2, values);
        const cg::CsrMatrix host_csr(2, 2, {0, 2, 4}, {0, 1, 0, 1}, values);

        for (const auto preconditioner : {cg::PreconditionerKind::none, cg::PreconditionerKind::jacobi})
        {
            for (const auto kernel : {cg::cuda::DenseKernel::native, cg::cuda::DenseKernel::cublas})
            {
                const cg::cuda::PreparedSolver dense(cg::cuda::DenseMatrix(host_dense.view(), context), context,
                                                     preconditioner, kernel);
                check_prepared_solver(dense, context);
            }

            const cg::cuda::PreparedSolver csr(cg::cuda::CsrMatrix(host_csr, context), context, preconditioner);
            check_prepared_solver(csr, context);
        }
        std::cout << "CUDA consumer passed (native/cuBLAS, host/device, dense/CSR)\n";
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "CUDA consumer failed: " << error.what() << '\n';
        return 1;
    }
}
