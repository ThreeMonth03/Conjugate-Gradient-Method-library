#pragma once

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <cg/matrix/dense_matrix_view.hpp>

namespace py = pybind11;

namespace cg
{
namespace python
{

void validate_vector_array(const py::array &value, bool writable);
cg::DenseMatrixView as_dense_matrix_view(const py::array &value);

} // namespace python
} // namespace cg
