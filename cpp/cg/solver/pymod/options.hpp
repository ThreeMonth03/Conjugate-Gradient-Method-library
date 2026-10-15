#pragma once

#include <pybind11/pybind11.h>

#include <cg/preconditioner/kind.hpp>
#include <cg/solver/options.hpp>

#include <string>
#include <utility>

namespace cg
{
namespace python
{

std::pair<std::size_t, std::size_t> parse_iteration_limits(pybind11::handle maxiter, pybind11::handle residual_refresh);
CgOptions parse_solver_options(double rtol, double atol, pybind11::handle maxiter, const std::string &algorithm,
                               pybind11::handle residual_refresh);
PreconditionerKind parse_preconditioner_kind(const std::string &name);

} // namespace python
} // namespace cg
