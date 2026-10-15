#include <cg/matrix/dense_matrix.hpp>
#include <cg/matrix/validation.hpp>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cg
{

DenseMatrixView::DenseMatrixView(const double *data, std::size_t rows, std::size_t cols, std::size_t row_stride)
    : data(data), rows(rows), cols(cols), row_stride(row_stride == 0 ? cols : row_stride)
{
    detail::validate(*this);
}

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols)
    : m_rows(rows), m_cols(cols), m_values(detail::checked_product(rows, cols))
{
}

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values)
    : m_rows(rows), m_cols(cols), m_values(std::move(values))
{
    if (m_values.size() != detail::checked_product(rows, cols))
    {
        throw std::invalid_argument("matrix shape does not match storage size");
    }
}

DenseMatrix::DenseMatrix(DenseMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_values(std::move(other.m_values))
{
}

DenseMatrix &DenseMatrix::operator=(DenseMatrix &&other) noexcept
{
    if (this != &other)
    {
        m_rows = std::exchange(other.m_rows, 0);
        m_cols = std::exchange(other.m_cols, 0);
        m_values = std::move(other.m_values);
    }
    return *this;
}

namespace detail
{
namespace
{
std::size_t matrix_span(DenseMatrixView view)
{
    if (view.rows == 0 || view.cols == 0)
    {
        return 0;
    }
    const auto preceding = checked_product(view.rows - 1, view.row_stride);
    if (preceding > std::numeric_limits<std::size_t>::max() - view.cols)
    {
        throw std::invalid_argument("matrix span overflows size_t");
    }
    return preceding + view.cols;
}

} // namespace
void validate(DenseMatrixView view)
{
    if (view.row_stride < view.cols)
    {
        throw std::invalid_argument("matrix row stride is smaller than its columns");
    }
    validate(ConstVectorView{view.data, matrix_span(view)});
}

bool overlaps(DenseMatrixView a, ConstVectorView b)
{
    validate(a);
    return overlaps(ConstVectorView{a.data, matrix_span(a)}, b);
}

} // namespace detail

} // namespace cg
