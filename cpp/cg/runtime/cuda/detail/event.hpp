#pragma once

#include <cg/runtime/cuda/detail/runtime.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

class Event
{
    public:
    explicit Event(int device) : m_device(device)
    {
        DeviceGuard guard(device);
        check(cudaEventCreateWithFlags(&m_value, cudaEventDisableTiming), "cudaEventCreateWithFlags");
    }
    Event(const Event &) = delete;
    Event &operator=(const Event &) = delete;
    ~Event()
    {
        on_device_noexcept(m_device, [&] { (void)cudaEventDestroy(m_value); });
    }
    void record(cudaStream_t producer) const
    {
        DeviceGuard guard(m_device);
        check(cudaEventRecord(m_value, producer), "cudaEventRecord");
    }
    void wait(cudaStream_t consumer) const
    {
        DeviceGuard guard(m_device);
        check(cudaStreamWaitEvent(consumer, m_value), "cudaStreamWaitEvent");
    }

    private:
    int m_device;
    cudaEvent_t m_value = nullptr;
};

} // namespace detail
} // namespace cuda
} // namespace cg
