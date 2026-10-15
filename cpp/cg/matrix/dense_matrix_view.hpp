#pragma once

#include <cstddef>

namespace cg
{

class DenseMatrixView
{
    public:
    DenseMatrixView() = default;
    DenseMatrixView(const double *data, std::size_t rows, std::size_t cols, std::size_t row_stride = 0);

    constexpr const double *data() const noexcept { return m_data; }
    constexpr std::size_t rows() const noexcept { return m_rows; }
    constexpr std::size_t cols() const noexcept { return m_cols; }
    constexpr std::size_t row_stride() const noexcept { return m_row_stride; }

    private:
    const double *m_data = nullptr;
    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    std::size_t m_row_stride = 0;
};

} // namespace cg
