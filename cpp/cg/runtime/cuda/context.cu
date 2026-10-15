#include <cg/runtime/cuda/context.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>

namespace cg
{
namespace cuda
{

int device_count() noexcept
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        (void)cudaGetLastError();
        return 0;
    }
    return count;
}
bool available() noexcept
{
    return device_count() > 0;
}
int runtime_version()
{
    int version = 0;
    detail::check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
    return version;
}
int driver_version()
{
    int version = 0;
    detail::check(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
    return version;
}
DeviceInfo device_info(int device)
{
    cudaDeviceProp properties{};
    detail::check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return {properties.name, properties.major, properties.minor, properties.totalGlobalMem};
}

Context::Context(int device, std::uintptr_t stream) : m_device(device), m_stream(stream)
{
    int count = 0;
    detail::check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (device < 0 || device >= count)
    {
        throw std::invalid_argument("CUDA device is out of range");
    }
    detail::DeviceGuard guard(device);
    int stream_device = -1;
    detail::check(cudaStreamGetDevice(detail::stream(*this), &stream_device), "cudaStreamGetDevice");
    if (stream_device != device)
    {
        throw std::invalid_argument("CUDA stream belongs to a different device");
    }
    unsigned int flags = 0;
    detail::check(cudaStreamGetFlags(detail::stream(*this), &flags), "cudaStreamGetFlags");
}
void Context::synchronize() const
{
    detail::DeviceGuard guard(m_device);
    detail::check(cudaStreamSynchronize(detail::stream(*this)), "cudaStreamSynchronize");
}

} // namespace cuda
} // namespace cg
