#pragma once

#include <cstddef>

namespace cg
{

enum class Status
{
    converged,
    max_iterations,
    non_positive_curvature,
    // Also covers underflow/overflow of recurrence products outside the usable
    // float64 squared-sum range, even when the reported stable norm is finite.
    non_finite
};

struct [[nodiscard("Check SolveInfo.status before using x")]] SolveInfo
{
    Status status = Status::max_iterations;
    std::size_t iterations = 0;
    std::size_t matvecs = 0;
    double residual_norm = 0.0;
    std::size_t restarts = 0;
};

const char *status_name(Status status) noexcept;

} // namespace cg
