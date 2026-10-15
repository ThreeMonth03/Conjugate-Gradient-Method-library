#pragma once

#include <cg/solver/options.hpp>

#include <cmath>
#include <stdexcept>

namespace cg
{
namespace detail
{

inline void validate_options(const CgOptions &options, bool has_preconditioner)
{
    if (!std::isfinite(options.rtol) || !std::isfinite(options.atol) || options.rtol < 0.0 || options.atol < 0.0)
    {
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    }
    if (options.algorithm != Algorithm::explicit_residual && options.algorithm != Algorithm::recurrence)
    {
        throw std::invalid_argument("unsupported CG algorithm");
    }
    if (has_preconditioner && options.algorithm != Algorithm::recurrence)
    {
        throw std::invalid_argument("Jacobi preconditioning requires recurrence CG");
    }
}

} // namespace detail
} // namespace cg
