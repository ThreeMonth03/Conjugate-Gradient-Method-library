#pragma once

#include <cg/backend/cuda/context.hpp>

#include <cstddef>
#include <memory>

namespace cg
{
namespace cuda
{
namespace detail
{

// Cleanup remembers the allocating device and never changes the caller's
// device selection permanently. Definitions live in the CUDA implementation.
struct DeviceDeleter
{
    int device = -1;
    void operator()(void *pointer) const noexcept;
};
template <class T> using DeviceAllocation = std::unique_ptr<T[], DeviceDeleter>;
void *allocate_device_storage(std::size_t count, std::size_t itemsize, int device);

template <class T> DeviceAllocation<T> device_allocation(std::size_t count, const Context &context)
{
    return DeviceAllocation<T>(static_cast<T *>(allocate_device_storage(count, sizeof(T), context.device())),
                               DeviceDeleter{context.device()});
}

} // namespace detail
} // namespace cuda
} // namespace cg
