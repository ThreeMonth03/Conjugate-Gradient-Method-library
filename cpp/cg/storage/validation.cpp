#include <cg/matrix/validation.hpp>
#include <cg/storage/validation.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cg
{
namespace detail
{

std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    {
        throw std::invalid_argument("storage size overflows size_t");
    }
    return a * b;
}

namespace
{
std::uintptr_t end_address(const double *data, std::size_t size)
{
    const auto bytes = checked_product(size, sizeof(double));
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("storage address range overflows uintptr_t");
    }
    return begin + bytes;
}

} // namespace
void validate(ConstVectorView view)
{
    if (view.size != 0 && view.data == nullptr)
    {
        throw std::invalid_argument("nonempty vector has a null data pointer");
    }
    if (view.size != 0 && reinterpret_cast<std::uintptr_t>(view.data) % alignof(double) != 0)
    {
        throw std::invalid_argument("float64 storage must be aligned for double access");
    }
    end_address(view.data, view.size);
}

bool overlaps(ConstVectorView a, ConstVectorView b)
{
    validate(a);
    validate(b);
    if (a.size == 0 || b.size == 0)
    {
        return false;
    }
    return reinterpret_cast<std::uintptr_t>(a.data) < end_address(b.data, b.size) &&
           reinterpret_cast<std::uintptr_t>(b.data) < end_address(a.data, a.size);
}

} // namespace detail
} // namespace cg
