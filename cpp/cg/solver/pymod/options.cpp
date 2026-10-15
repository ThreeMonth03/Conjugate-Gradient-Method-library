#include <cg/solver/pymod/options.hpp>

#include <stdexcept>

namespace cg
{
namespace python
{

CgOptions parse_solver_options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                               std::size_t residual_refresh)
{
    CgOptions result;
    result.rtol = rtol;
    result.atol = atol;
    result.maxiter = maxiter;
    result.residual_refresh = residual_refresh;
    if (algorithm == "standard")
    {
        result.algorithm = Algorithm::recurrence;
    }
    else if (algorithm == "explicit")
    {
        result.algorithm = Algorithm::explicit_residual;
    }
    else
    {
        throw std::invalid_argument("algorithm must be 'standard' or 'explicit'");
    }
    return result;
}

PreconditionerKind parse_preconditioner_kind(const std::string &name)
{
    if (name == "none")
    {
        return PreconditionerKind::none;
    }
    if (name == "jacobi")
    {
        return PreconditionerKind::jacobi;
    }
    throw std::invalid_argument("preconditioner must be 'none' or 'jacobi'");
}

} // namespace python
} // namespace cg
