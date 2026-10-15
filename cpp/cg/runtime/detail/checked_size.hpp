#pragma once

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cg
{
namespace detail
{

constexpr std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    {
        throw std::invalid_argument("storage size overflows size_t");
    }
    return a * b;
}

} // namespace detail
} // namespace cg
