#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/runtime/detail/checked_size.hpp>
#include <cg/storage/cuda/device_memory.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void *allocate_device_storage(std::size_t count, std::size_t itemsize, int device)
{
    const auto bytes = cg::detail::checked_product(count, itemsize);
    DeviceGuard guard(device);
    void *pointer = nullptr;
    if (bytes != 0)
    {
        check(cudaMalloc(&pointer, bytes), "cudaMalloc");
    }
    return pointer;
}
void DeviceDeleter::operator()(void *pointer) const noexcept
{
    if (pointer != nullptr)
    {
        on_device_noexcept(device, [&] { (void)cudaFree(pointer); });
    }
}

} // namespace detail
} // namespace cuda
} // namespace cg
