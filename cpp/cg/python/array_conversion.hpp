#pragma once

#include <pybind11/pybind11.h>

#include <pybind11/numpy.h>
#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/storage/vector_view.hpp>

namespace py = pybind11;

namespace cg
{
namespace python
{

cg::ConstVectorView as_const_vector_view(const py::array &value);
cg::VectorView as_mutable_vector_view(py::array &value);
cg::DenseMatrixView as_dense_matrix_view(const py::array &value);

} // namespace python
} // namespace cg
