#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace cg
{
namespace cuda
{

bool available() noexcept;
int device_count() noexcept;
int runtime_version();
int driver_version();

struct DeviceInfo
{
    std::string name;
    int major = 0;
    int minor = 0;
    std::size_t total_memory = 0;
};
DeviceInfo device_info(int device);

// The stream is borrowed, must belong to device, and must outlive every use.
// CUDA operations temporarily select this device on the calling thread and
// restore the previous selection. No process-wide stream setting is changed.
// Borrowed input data must be ready on this stream before submitting a solve.
class Context
{
    public:
    explicit Context(int device = 0, std::uintptr_t stream = 0);
    int device() const noexcept { return m_device; }
    std::uintptr_t stream() const noexcept { return m_stream; }
    void synchronize() const;

    private:
    int m_device;
    std::uintptr_t m_stream;
};

} // namespace cuda
} // namespace cg
