#pragma once

#include <cg/solver/options.hpp>
#include <cg/solver/solve_info.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>

namespace cg
{
namespace detail
{

inline bool unusable_squared_scalar(double value)
{
    return !std::isfinite(value) || value < std::numeric_limits<double>::min();
}

struct InitialResidualDecision
{
    double tolerance = 0.0;
    std::optional<Status> status;
};

inline InitialResidualDecision initial_residual_decision(double b_norm, double residual_norm, double rho,
                                                         const CgOptions &options)
{
    if (!std::isfinite(b_norm) || !std::isfinite(residual_norm))
    {
        return {0.0, Status::non_finite};
    }
    const double tolerance = std::max(options.atol, options.rtol * b_norm);
    // An actual converged norm and a zero iteration budget take precedence
    // over the usable range of squared products needed by an iteration.
    if (residual_norm <= tolerance)
    {
        return {tolerance, Status::converged};
    }
    if (options.maxiter == 0)
    {
        return {tolerance, Status::max_iterations};
    }
    if (unusable_squared_scalar(rho))
    {
        return {tolerance, Status::non_finite};
    }
    return {tolerance, std::nullopt};
}

inline Status true_residual_status(Status requested, double residual_norm, double tolerance)
{
    if (!std::isfinite(residual_norm) || requested == Status::non_finite)
    {
        return Status::non_finite;
    }
    return residual_norm <= tolerance ? Status::converged : requested;
}

// No terminal status means the verified residual must replace the recurrence
// residual and restart the direction. Backend work stays in each engine.
inline std::optional<Status> residual_checkpoint_status(double residual_norm, double rho, double tolerance, bool last)
{
    if (!std::isfinite(residual_norm))
    {
        return Status::non_finite;
    }
    if (residual_norm <= tolerance)
    {
        return Status::converged;
    }
    if (unusable_squared_scalar(rho))
    {
        return Status::non_finite;
    }
    if (last)
    {
        return Status::max_iterations;
    }
    return std::nullopt;
}

} // namespace detail
} // namespace cg
