#include <pybind11/pybind11.h>

#include <cg/python/module.hpp>

PYBIND11_MODULE(_core, module)
{
    module.doc() = "Synchronous CPU CG kernels with caller-owned output and reusable workspace";
    cg::python::register_cpu_matrix(module);
    cg::python::register_cpu_solver(module);
    cg::python::register_cpu_runtime(module);
}
