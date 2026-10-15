#include <cg/storage/cuda/detail/allocation.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void *allocate_device_storage(std::size_t count, std::size_t itemsize, int device)
{
    const auto bytes = checked_product(count, itemsize);
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
PinnedScalar pinned_scalar()
{
    double *pointer = nullptr;
    check(cudaMallocHost(reinterpret_cast<void **>(&pointer), sizeof(double)), "cudaMallocHost");
    return PinnedScalar(pointer);
}
void PinnedDeleter::operator()(double *pointer) const noexcept
{
    (void)cudaFreeHost(pointer);
}

} // namespace detail
} // namespace cuda
} // namespace cg
