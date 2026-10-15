#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/runtime/detail/checked_size.hpp>
#include <cg/storage/cuda/detail/validation.hpp>

#include <cstdint>
#include <limits>

namespace cg
{
namespace cuda
{

namespace detail
{

void validate_pointer(const void *data, std::size_t bytes, std::size_t alignment, int device, const Context &context,
                      bool inspect_pointer)
{
    if (bytes == 0)
    {
        return;
    }
    if (device != context.device())
    {
        throw std::invalid_argument("CUDA storage belongs to a different device");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || begin % alignment != 0 || begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("invalid CUDA storage pointer, alignment or span");
    }
    if (inspect_pointer)
    {
        DeviceGuard guard(device);
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, data);
        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            throw std::invalid_argument("expected CUDA device storage");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        {
            throw std::invalid_argument("expected storage on the selected CUDA device");
        }
    }
}
void validate(ConstVectorView view, const Context &context, bool inspect_pointer)
{
    validate_pointer(view.data(), cg::detail::checked_product(view.size(), sizeof(double)), alignof(double),
                     view.device(), context, inspect_pointer);
}
bool overlaps(ConstVectorView a, ConstVectorView b)
{
    if (a.size() == 0 || b.size() == 0 || a.device() != b.device())
    {
        return false;
    }
    const auto abegin = reinterpret_cast<std::uintptr_t>(a.data());
    const auto bbegin = reinterpret_cast<std::uintptr_t>(b.data());
    const auto abytes = cg::detail::checked_product(a.size(), sizeof(double)),
               bbytes = cg::detail::checked_product(b.size(), sizeof(double));
    if (abegin > std::numeric_limits<std::uintptr_t>::max() - abytes ||
        bbegin > std::numeric_limits<std::uintptr_t>::max() - bbytes)
    {
        throw std::invalid_argument("CUDA storage address span overflows uintptr_t");
    }
    return abegin < bbegin + bbytes && bbegin < abegin + abytes;
}

} // namespace detail

} // namespace cuda
} // namespace cg
