#pragma once

#include <cstddef>

namespace cg
{
namespace cuda
{

struct DenseMatrixView
{
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_stride = 0;
    int device = -1;
};

} // namespace cuda
} // namespace cg
