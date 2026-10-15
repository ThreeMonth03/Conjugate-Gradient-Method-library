#include <cg/matrix/csr_matrix.hpp>
#include <cg/matrix/detail/csr_validation.hpp>
#include <cg/storage/detail/validation.hpp>

#include <limits>
#include <stdexcept>
#include <utility>

namespace cg
{

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> indptr,
                     std::vector<std::size_t> indices, std::vector<double> values)
    : m_rows(rows), m_cols(cols), m_indptr(std::move(indptr)), m_indices(std::move(indices)),
      m_values(std::move(values))
{
    detail::validate_csr_structure(rows, cols, m_indptr, m_indices, m_values.size());
}

CsrMatrix::CsrMatrix(CsrMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_indptr(std::move(other.m_indptr)), m_indices(std::move(other.m_indices)), m_values(std::move(other.m_values))
{
}

CsrMatrix &CsrMatrix::operator=(const CsrMatrix &other)
{
    if (this != &other)
    {
        CsrMatrix copy(other);
        swap(copy);
    }
    return *this;
}

void CsrMatrix::swap(CsrMatrix &other) noexcept
{
    std::swap(m_rows, other.m_rows);
    std::swap(m_cols, other.m_cols);
    m_indptr.swap(other.m_indptr);
    m_indices.swap(other.m_indices);
    m_values.swap(other.m_values);
}

CsrMatrix &CsrMatrix::operator=(CsrMatrix &&other) noexcept
{
    if (this != &other)
    {
        m_rows = std::exchange(other.m_rows, 0);
        m_cols = std::exchange(other.m_cols, 0);
        m_indptr = std::move(other.m_indptr);
        m_indices = std::move(other.m_indices);
        m_values = std::move(other.m_values);
    }
    return *this;
}

namespace detail
{

void validate_csr_structure(std::size_t rows, std::size_t cols, std::span<const std::size_t> indptr,
                            std::span<const std::size_t> indices, std::size_t nnz)
{
    if (rows == std::numeric_limits<std::size_t>::max() || indptr.size() != rows + 1 || indices.size() != nnz ||
        indptr.front() != 0 || indptr.back() != nnz)
    {
        throw std::invalid_argument("invalid CSR storage dimensions or endpoints");
    }
    for (std::size_t row = 0; row < rows; ++row)
    {
        const auto begin = indptr[row];
        const auto end = indptr[row + 1];
        if (begin > end || end > nnz)
        {
            throw std::invalid_argument("CSR row pointers must be monotone and within storage");
        }
        for (std::size_t entry = begin; entry < end; ++entry)
        {
            if (indices[entry] >= cols || (entry != begin && indices[entry - 1] >= indices[entry]))
            {
                throw std::invalid_argument("CSR columns must be in range, sorted, and unique");
            }
        }
    }
}

bool overlaps(const CsrMatrix &A, ConstVectorView x)
{
    return overlaps(A.values(), x);
}

} // namespace detail

} // namespace cg
