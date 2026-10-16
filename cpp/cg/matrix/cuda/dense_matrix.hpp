#pragma once

#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/storage/cuda/vector.hpp>

namespace cg
{
namespace cuda
{

class DenseMatrix
{
    public:
    DenseMatrix(cg::DenseMatrixView host, const Context &context);
    DenseMatrix(DenseMatrixView source, const Context &context);
    DenseMatrix(const DenseMatrix &) = delete;
    DenseMatrix &operator=(const DenseMatrix &) = delete;
    DenseMatrix(DenseMatrix &&other) noexcept;
    DenseMatrix &operator=(DenseMatrix &&other) noexcept;
    std::size_t rows() const noexcept { return m_rows; }
    std::size_t cols() const noexcept { return m_cols; }
    int device() const noexcept { return m_values.device(); }
    DenseMatrixView view() const & noexcept { return {m_values.data(), m_rows, m_cols, m_cols, m_values.device()}; }
    DenseMatrixView view() const && = delete;

    private:
    std::size_t m_rows;
    std::size_t m_cols;
    Vector m_values;
};

} // namespace cuda
} // namespace cg
