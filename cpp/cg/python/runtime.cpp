#include <pybind11/pybind11.h>

#include <cg/backend/cpu/backend.hpp>

#include <cg/python/module.hpp>

namespace py = pybind11;

namespace cg
{
namespace python
{

void register_cpu_runtime(py::module_ &m)
{
    m.def("openmp_enabled", &cg::CpuBackend::openmp_enabled, R"doc(
Report OpenMP support in the extension imported by this process.

Returns
-------
bool
    Whether native kernel calls can request more than one CPU thread.
)doc");
    m.def("blas_enabled", &cg::CpuBackend::blas_enabled,
          R"doc(
Report optional sequential BLAS support in the imported extension.

Returns
-------
bool
    Whether ``dense_kernel='blas'`` is available for dense, one-thread calls.

Notes
-----
This reports the actual imported binary. It does not change kernel selection
or inspect another build directory. Native remains the default kernel.
)doc");
    m.def(
        "build_info",
        []
        {
            py::dict result;
            result["compiler"] = CG_COMPILER;
            result["build_type"] = CG_BUILD_TYPE;
            result["flags"] = CG_BUILD_FLAGS;
            result["openmp"] = cg::CpuBackend::openmp_enabled();
            py::dict blas;
            const bool enabled = cg::CpuBackend::blas_enabled();
            blas["enabled"] = enabled;
            blas["configuration"] = cg::CpuBackend::blas_info();
            if (enabled)
            {
                blas["provider"] = "OpenBLAS";
                blas["integer_bits"] = 32;
                blas["thread_mode"] = "sequential";
                blas["symbol_prefix"] = "cgpy_";
            }
            else
            {
                blas["provider"] = py::none();
                blas["integer_bits"] = py::none();
                blas["thread_mode"] = py::none();
                blas["symbol_prefix"] = py::none();
            }
            result["blas"] = blas;
            return result;
        },
        R"doc(
Describe the native extension imported by this Python process.

Returns
-------
dict
    Compiler, build type, flags and OpenMP support. The ``blas`` dictionary
    records ``enabled``, provider/configuration, integer width, thread mode
    and symbol prefix. Provider-specific fields are None without BLAS.

Notes
-----
BLAS-enabled builds use the private sequential LP64 OpenBLAS provider. Build
information belongs to this loaded binary, not another preset or installed wheel.
)doc");
}

} // namespace python
} // namespace cg
