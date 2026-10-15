#pragma once

#include <cg/backend/cuda/detail/runtime.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

template <class T> class DeviceAllocation
{
    public:
    DeviceAllocation(std::size_t count, const Context &context) : m_device(context.device())
    {
        const auto bytes = product(count, sizeof(T));
        DeviceGuard guard(m_device);
        if (bytes != 0)
        {
            check(cudaMalloc(reinterpret_cast<void **>(&m_data), bytes), "cudaMalloc");
        }
    }
    ~DeviceAllocation()
    {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(m_device);
        (void)cudaFree(m_data);
        (void)cudaSetDevice(previous);
    }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    T *data() noexcept { return m_data; }
    const T *data() const noexcept { return m_data; }

    private:
    int m_device;
    T *m_data = nullptr;
};

struct PinnedScalar
{
    PinnedScalar() { check(cudaMallocHost(reinterpret_cast<void **>(&data), sizeof(double)), "cudaMallocHost"); }
    PinnedScalar(const PinnedScalar &) = delete;
    PinnedScalar &operator=(const PinnedScalar &) = delete;
    ~PinnedScalar() { (void)cudaFreeHost(data); }
    double *data = nullptr;
};

} // namespace detail
} // namespace cuda
} // namespace cg
