#include <cg/solver/pymod/options.hpp>

#include <stdexcept>

namespace py = pybind11;

namespace cg
{
namespace python
{

namespace
{
std::size_t parse_iteration_count(py::handle value, const char *name)
{
    auto index = py::reinterpret_steal<py::object>(PyNumber_Index(value.ptr()));
    if (!index)
    {
        if (PyErr_ExceptionMatches(PyExc_TypeError))
        {
            PyErr_Clear();
            throw py::value_error(std::string(name) + " must be a nonnegative integer");
        }
        throw py::error_already_set();
    }
    const auto count = PyLong_AsSize_t(index.ptr());
    if (PyErr_Occurred())
    {
        if (PyErr_ExceptionMatches(PyExc_OverflowError))
        {
            PyErr_Clear();
            throw py::value_error(std::string(name) + " must fit a nonnegative size_t");
        }
        throw py::error_already_set();
    }
    return count;
}
} // namespace

std::pair<std::size_t, std::size_t> parse_iteration_limits(py::handle maxiter, py::handle residual_refresh)
{
    return {parse_iteration_count(maxiter, "maxiter"), parse_iteration_count(residual_refresh, "residual_refresh")};
}

CgOptions parse_solver_options(double rtol, double atol, py::handle maxiter, const std::string &algorithm,
                               py::handle residual_refresh)
{
    const auto [iterations, refresh] = parse_iteration_limits(maxiter, residual_refresh);
    CgOptions result;
    result.rtol = rtol;
    result.atol = atol;
    result.maxiter = iterations;
    result.residual_refresh = refresh;
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
