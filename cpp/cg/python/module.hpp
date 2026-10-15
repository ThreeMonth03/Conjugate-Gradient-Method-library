#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace cg
{
namespace python
{

void register_cpu_matrix(py::module_ &module);
void register_cpu_solver(py::module_ &module);
void register_cpu_runtime(py::module_ &module);
void register_cuda_storage(py::module_ &module);
void register_cuda_matrix(py::module_ &module);
void register_cuda_solver(py::module_ &module);
void register_cuda_runtime(py::module_ &module);

} // namespace python
} // namespace cg
