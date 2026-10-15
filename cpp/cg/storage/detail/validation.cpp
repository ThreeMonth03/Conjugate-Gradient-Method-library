#include <cg/runtime/detail/checked_size.hpp>
#include <cg/storage/detail/validation.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cg
{
namespace detail
{

namespace
{
std::uintptr_t end_address(const double *data, std::size_t size)
{
    const auto bytes = cg::detail::checked_product(size, sizeof(double));
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("storage address range overflows uintptr_t");
    }
    return begin + bytes;
}

} // namespace
void validate_raw_vector(const double *data, std::size_t size)
{
    if (size != 0 && data == nullptr)
    {
        throw std::invalid_argument("nonempty vector has a null data pointer");
    }
    if (size != 0 && reinterpret_cast<std::uintptr_t>(data) % alignof(double) != 0)
    {
        throw std::invalid_argument("float64 storage must be aligned for double access");
    }
    end_address(data, size);
}

void validate(ConstVectorView view)
{
    validate_raw_vector(view.data(), view.size());
}

bool overlaps_raw(const double *a, std::size_t a_size, const double *b, std::size_t b_size)
{
    validate_raw_vector(a, a_size);
    validate_raw_vector(b, b_size);
    if (a_size == 0 || b_size == 0)
    {
        return false;
    }
    return reinterpret_cast<std::uintptr_t>(a) < end_address(b, b_size) &&
           reinterpret_cast<std::uintptr_t>(b) < end_address(a, a_size);
}

bool overlaps(ConstVectorView a, ConstVectorView b)
{
    return overlaps_raw(a.data(), a.size(), b.data(), b.size());
}

} // namespace detail
} // namespace cg
