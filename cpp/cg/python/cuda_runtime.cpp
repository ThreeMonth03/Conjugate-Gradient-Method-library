#include <pybind11/pybind11.h>

#include <cg/python/cuda_array_conversion.hpp>
#include <cg/python/module.hpp>

#include <iomanip>
#include <sstream>

namespace cg
{
namespace python
{

void register_cuda_runtime(py::module_ &m)
{
    m.def(
        "is_available", [](int device) { return device >= 0 && device < gpu::device_count(); }, py::arg("device") = 0);
    m.def(
        "synchronize", [](int device, std::uintptr_t stream) { gpu::Context(device, stream).synchronize(); },
        py::arg("device") = 0, py::arg("stream") = 0);
    m.def(
        "runtime_info",
        [](int device)
        {
            gpu::detail::DeviceGuard selected(device);
            cudaDeviceProp property{};
            check(cudaGetDeviceProperties(&property, device));
            std::ostringstream uuid;
            uuid << "GPU-" << std::hex << std::setfill('0');
            for (std::size_t i = 0; i != 16; ++i)
            {
                if (i == 4 || i == 6 || i == 8 || i == 10)
                {
                    uuid << '-';
                }
                uuid << std::setw(2) << static_cast<unsigned>(static_cast<unsigned char>(property.uuid.bytes[i]));
            }
            py::dict result;
            result["device"] = device;
            result["name"] = property.name;
            result["uuid"] = uuid.str();
            result["compute_capability"] = py::make_tuple(property.major, property.minor);
            result["total_memory"] = property.totalGlobalMem;
            result["runtime_version"] = gpu::runtime_version();
            result["driver_api_version"] = gpu::driver_version();
            return result;
        },
        py::arg("device") = 0);
    m.def("build_info",
          []
          {
              py::dict result;
              result["enabled"] = true;
              result["toolkit_version"] = CG_CUDA_TOOLKIT_VERSION;
              result["architectures"] = CG_CUDA_ARCHITECTURES;
              result["compiler"] = CG_CUDA_COMPILER;
              result["host_compiler"] = CG_CUDA_HOST_COMPILER;
              result["build_type"] = CG_CUDA_BUILD_TYPE;
              result["flags"] = CG_CUDA_BUILD_FLAGS;
              result["scalar_dtype"] = "float64";
              return result;
          });
}

} // namespace python
} // namespace cg
