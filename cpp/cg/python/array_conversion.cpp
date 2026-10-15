#include <pybind11/pybind11.h>

#include <cg/python/array_conversion.hpp>
#include <cg/storage/validation.hpp>

#include <stdexcept>

namespace cg
{
namespace python
{

void vector_array(const py::array &value, bool writable)
{
    if (!value.dtype().is(py::dtype::of<double>()) || value.ndim() != 1 || !(value.flags() & py::array::c_style))
    {
        throw std::invalid_argument("Expected a contiguous one-dimensional float64 array");
    }
    if (writable && !value.writeable())
    {
        throw std::invalid_argument("Output array must be writable");
    }
    detail::validate_raw_vector(static_cast<const double *>(value.data()), static_cast<std::size_t>(value.size()));
}

cg::DenseMatrixView matrix_view(const py::array &value)
{
    if (!value.dtype().is(py::dtype::of<double>()) || value.ndim() != 2 || !(value.flags() & py::array::c_style))
    {
        throw std::invalid_argument("Expected a C-contiguous two-dimensional float64 matrix");
    }
    return {static_cast<const double *>(value.data()), static_cast<std::size_t>(value.shape(0)),
            static_cast<std::size_t>(value.shape(1))};
}

cg::CgOptions options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                      std::size_t residual_refresh)
{
    cg::CgOptions result;
    result.rtol = rtol;
    result.atol = atol;
    result.maxiter = maxiter;
    result.residual_refresh = residual_refresh;
    if (algorithm == "standard")
    {
        result.algorithm = cg::Algorithm::recurrence;
    }
    else if (algorithm == "explicit")
    {
        result.algorithm = cg::Algorithm::explicit_residual;
    }
    else
    {
        throw std::invalid_argument("algorithm must be 'standard' or 'explicit'");
    }
    return result;
}

} // namespace python
} // namespace cg
