#pragma once

#include <cg/runtime/detail/checked_size.hpp>

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace cg
{
namespace detail
{

constexpr std::size_t storage_extent(std::size_t rows, std::size_t cols, std::size_t row_stride)
{
    if (row_stride < cols)
    {
        throw std::invalid_argument("matrix row stride is smaller than its columns");
    }
    if (rows == 0 || cols == 0)
    {
        return 0;
    }
    const auto preceding = cg::detail::checked_product(rows - 1, row_stride);
    if (preceding > std::numeric_limits<std::size_t>::max() - cols)
    {
        throw std::invalid_argument("matrix span overflows size_t");
    }
    return preceding + cols;
}

} // namespace detail
} // namespace cg
