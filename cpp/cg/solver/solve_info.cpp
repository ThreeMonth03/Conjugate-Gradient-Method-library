#include <cg/solver/solve_info.hpp>

namespace cg
{

const char *status_name(Status status) noexcept
{
    switch (status)
    {
    case Status::converged:
        return "converged";
    case Status::max_iterations:
        return "max_iterations";
    case Status::non_positive_curvature:
        return "non_positive_curvature";
    case Status::non_finite:
        return "non_finite";
    }
    return "unknown";
}

} // namespace cg
