#pragma once

#include <cstddef>
#include <span>

namespace cg
{
namespace detail
{

void validate_csr_structure(std::size_t rows, std::size_t cols, std::span<const std::size_t> indptr,
                            std::span<const std::size_t> indices, std::size_t nnz);

} // namespace detail
} // namespace cg
