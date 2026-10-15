#pragma once

#include <cg/runtime/cuda/context.hpp>

#include <cuda_runtime.h>

#include <limits>
#include <stdexcept>
#include <string>

namespace cg
{
namespace cuda
{
namespace detail
{

inline void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
    {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}
class DeviceGuard
{
    public:
    explicit DeviceGuard(int device)
    {
        check(cudaGetDevice(&m_previous), "cudaGetDevice");
        if (m_previous != device)
        {
            check(cudaSetDevice(device), "cudaSetDevice");
        }
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    ~DeviceGuard() { (void)cudaSetDevice(m_previous); }

    private:
    int m_previous = 0;
};
inline cudaStream_t stream(const Context &context)
{
    return reinterpret_cast<cudaStream_t>(context.stream());
}
template <class Cleanup> void on_device_noexcept(int device, Cleanup cleanup) noexcept
{
    int previous = 0;
    const bool restore = cudaGetDevice(&previous) == cudaSuccess;
    (void)cudaSetDevice(device);
    cleanup();
    if (restore)
    {
        (void)cudaSetDevice(previous);
    }
}
inline std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    {
        throw std::invalid_argument("CUDA storage size overflows size_t");
    }
    return a * b;
}

} // namespace detail
} // namespace cuda
} // namespace cg
