#include <cg/matrix/dense_matrix.hpp>
#include <cg/matrix/detail/storage_extent.hpp>
#include <cg/matrix/detail/validation.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <stdexcept>
#include <utility>

namespace cg
{

DenseMatrixView::DenseMatrixView(const double *data, std::size_t rows, std::size_t cols, std::size_t row_stride)
    : m_data(data), m_rows(rows), m_cols(cols), m_row_stride(row_stride == 0 ? cols : row_stride)
{
    detail::validate(*this);
}

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols)
    : m_rows(rows), m_cols(cols), m_values(cg::detail::checked_product(rows, cols))
{
}

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values)
    : m_rows(rows), m_cols(cols), m_values(std::move(values))
{
    if (m_values.size() != cg::detail::checked_product(rows, cols))
    {
        throw std::invalid_argument("matrix shape does not match storage size");
    }
}

DenseMatrix::DenseMatrix(DenseMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_values(std::move(other.m_values))
{
}

DenseMatrix &DenseMatrix::operator=(const DenseMatrix &other)
{
    if (this != &other)
    {
        DenseMatrix copy(other);
        swap(copy);
    }
    return *this;
}

void DenseMatrix::swap(DenseMatrix &other) noexcept
{
    std::swap(m_rows, other.m_rows);
    std::swap(m_cols, other.m_cols);
    m_values.swap(other.m_values);
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
void validate(DenseMatrixView view)
{
    validate_raw_vector(view.data(), cg::detail::storage_extent(view.rows(), view.cols(), view.row_stride()));
}

bool overlaps(DenseMatrixView a, ConstVectorView b)
{
    validate(a);
    return overlaps_raw(a.data(), cg::detail::storage_extent(a.rows(), a.cols(), a.row_stride()), b.data(), b.size());
}

} // namespace detail

} // namespace cg
