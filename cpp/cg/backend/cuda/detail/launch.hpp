#pragma once

#include <cg/backend/cuda/detail/runtime.hpp>

#include <algorithm>
#include <cstddef>

namespace cg
{
namespace cuda
{
namespace detail
{

inline void launch_check()
{
    check(cudaGetLastError(), "CUDA kernel launch");
}
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

inline unsigned int dense_blocks(std::size_t rows)
{
    return static_cast<unsigned int>(std::min<std::size_t>(std::max<std::size_t>(rows, 1), max_blocks));
}

} // namespace detail
} // namespace cuda
} // namespace cg
