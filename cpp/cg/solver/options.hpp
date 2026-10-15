#pragma once

#include <cstddef>

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
    // Also covers underflow/overflow of recurrence products outside the usable
    // float64 squared-sum range, even when the reported stable norm is finite.
    non_finite
};

struct CgOptions
{
    double rtol = 1e-8;
    double atol = 0.0;
    std::size_t maxiter = 1000;
    Algorithm algorithm = Algorithm::recurrence;
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

} // namespace cg
