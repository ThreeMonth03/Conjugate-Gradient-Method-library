#pragma once

#include <algorithm>
#include <cstddef>

namespace cg
{
namespace cuda
{
namespace detail
{

constexpr unsigned int block_size = 256;
constexpr unsigned int max_blocks = 256;
inline unsigned int blocks(std::size_t size)
{
    if (size == 0)
    {
        return 1;
    }
    return static_cast<unsigned int>(std::min<std::size_t>((size - 1) / block_size + 1, max_blocks));
}

} // namespace detail
} // namespace cuda
} // namespace cg
