#include <pybind11/pybind11.h>

#include <cg/python/module.hpp>

PYBIND11_MODULE(_cuda, module)
{
    py::module_::import("cgpy._core");
    module.doc() = "Optional synchronous CUDA CG with explicit device storage and streams";
    cg::python::register_cuda_runtime(module);
    cg::python::register_cuda_storage(module);
    cg::python::register_cuda_matrix(module);
    cg::python::register_cuda_solver(module);
}
