#pragma once

#include <cg/matrix/dense_matrix_view.hpp>

#include <vector>

namespace cg
{

class DenseMatrix
{
    public:
    DenseMatrix(std::size_t rows, std::size_t cols);
    DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values);
    DenseMatrix(const DenseMatrix &) = default;
    DenseMatrix &operator=(const DenseMatrix &) = default;
    DenseMatrix(DenseMatrix &&other) noexcept;
    DenseMatrix &operator=(DenseMatrix &&other) noexcept;
    std::size_t rows() const noexcept { return m_rows; }
    std::size_t cols() const noexcept { return m_cols; }
    double *data() noexcept { return m_values.data(); }
    const double *data() const noexcept { return m_values.data(); }
    DenseMatrixView view() const { return {data(), m_rows, m_cols}; }

    private:
    std::size_t m_rows;
    std::size_t m_cols;
    std::vector<double> m_values;
};

} // namespace cg
