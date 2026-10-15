#pragma once

#include <cstddef>

namespace cg
{
namespace cuda
{

enum class IndexType
{
    int32,
    int64
};
struct CsrMatrixView
{
    const void *indptr = nullptr;
    const void *indices = nullptr;
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz = 0;
    IndexType index_type = IndexType::int64;
    int device = -1;
};

} // namespace cuda
} // namespace cg
