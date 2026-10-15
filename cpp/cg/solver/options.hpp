#pragma once

#include <cstddef>

namespace cg
{

enum class Algorithm
{
    explicit_residual,
    recurrence
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

} // namespace cg
