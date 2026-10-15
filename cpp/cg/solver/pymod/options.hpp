#pragma once

#include <cg/preconditioner/kind.hpp>
#include <cg/solver/options.hpp>

#include <string>

namespace cg
{
namespace python
{

CgOptions parse_solver_options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                               std::size_t residual_refresh);
PreconditionerKind parse_preconditioner_kind(const std::string &name);

} // namespace python
} // namespace cg
