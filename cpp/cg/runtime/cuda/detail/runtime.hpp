#pragma once

#include <cg/runtime/cuda/context.hpp>

#include <cuda_runtime.h>

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
            m_switched = true;
        }
    }
    DeviceGuard(const DeviceGuard &) = delete;
    DeviceGuard &operator=(const DeviceGuard &) = delete;
    ~DeviceGuard() noexcept
    {
        // A guarded callback can change the current device even when entry
        // needed no switch. Skip restoration only after verifying it is still
        // unnecessary; a failed query retains the old best-effort cleanup.
        int current = 0;
        if (m_switched || cudaGetDevice(&current) != cudaSuccess || current != m_previous)
        {
            (void)cudaSetDevice(m_previous);
        }
    }

    private:
    int m_previous = 0;
    bool m_switched = false;
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
} // namespace detail
} // namespace cuda
} // namespace cg
