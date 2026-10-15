#pragma once

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/solver/options.hpp>

#include <string>

namespace py = pybind11;

namespace cg
{
namespace python
{

void vector_array(const py::array &value, bool writable);
cg::DenseMatrixView matrix_view(const py::array &value);
cg::CgOptions options(double rtol, double atol, std::size_t maxiter, const std::string &algorithm,
                      std::size_t residual_refresh);

} // namespace python
} // namespace cg
