#include <cg/matrix/csr_matrix.hpp>
#include <cg/matrix/validation.hpp>
#include <cg/storage/validation.hpp>

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
    if (rows == std::numeric_limits<std::size_t>::max() || m_indptr.size() != rows + 1 ||
        m_indices.size() != m_values.size() || m_indptr.front() != 0 || m_indptr.back() != m_values.size())
    {
        throw std::invalid_argument("invalid CSR storage dimensions or endpoints");
    }
    for (std::size_t row = 0; row < rows; ++row)
    {
        const auto begin = m_indptr[row];
        const auto end = m_indptr[row + 1];
        if (begin > end || end > m_values.size())
        {
            throw std::invalid_argument("CSR row pointers must be monotone and within storage");
        }
        for (std::size_t entry = begin; entry < end; ++entry)
        {
            if (m_indices[entry] >= cols || (entry != begin && m_indices[entry - 1] >= m_indices[entry]))
            {
                throw std::invalid_argument("CSR columns must be in range, sorted, and unique");
            }
        }
    }
}

CsrMatrix::CsrMatrix(CsrMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_indptr(std::move(other.m_indptr)), m_indices(std::move(other.m_indices)), m_values(std::move(other.m_values))
{
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

bool overlaps(const CsrMatrix &A, ConstVectorView x)
{
    return overlaps(ConstVectorView{A.values().data(), A.nnz()}, x);
}

} // namespace detail

} // namespace cg
