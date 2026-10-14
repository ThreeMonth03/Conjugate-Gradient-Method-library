#ifndef CG_SOLVER_HPP
#define CG_SOLVER_HPP

#include "cg/backend.hpp"

namespace cg
{

enum class Algorithm
{
    explicit_residual,
    recurrence
};
enum class Status
{
    converged,
    max_iterations,
    non_positive_curvature,
    non_finite
};

struct CgOptions
{
    double rtol = 1e-8;
    double atol = 0.0;
    std::size_t maxiter = 1000;
    Algorithm algorithm = Algorithm::explicit_residual;
    // Zero disables periodic replacement. Candidate convergence and the final
    // result always use the true residual, even with replacement disabled.
    std::size_t residual_refresh = 50;
};

struct SolveInfo
{
    Status status = Status::max_iterations;
    std::size_t iterations = 0;
    std::size_t matvecs = 0;
    double residual_norm = 0.0;
    std::size_t restarts = 0;
};

const char *status_name(Status status) noexcept;

// x is an initial guess and output. The output and workspace must be disjoint
// from A and b. The workspace must match A.rows; prepare it outside timing.
// Inputs are validated once, and no vector storage is allocated by the solve.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options = {});
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options = {});

} // namespace cg

#endif
