#pragma once

#include <cstddef>

namespace cg
{

struct DenseMatrixView
{
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_stride = 0;

    DenseMatrixView() = default;
    DenseMatrixView(const double *data, std::size_t rows, std::size_t cols, std::size_t row_stride = 0);
};

} // namespace cg
