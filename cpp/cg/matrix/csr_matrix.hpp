#pragma once

#include <cg/storage/vector_view.hpp>

#include <vector>

namespace cg
{

// Canonical owning CSR: row pointers are monotone and column indices within
// each row are strictly increasing. Construction rejects duplicates and
// unsorted columns rather than silently changing the supplied matrix.
class CsrMatrix
{
    public:
    CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> indptr, std::vector<std::size_t> indices,
              std::vector<double> values);
    CsrMatrix(const CsrMatrix &) = default;
    CsrMatrix &operator=(const CsrMatrix &other);
    CsrMatrix(CsrMatrix &&other) noexcept;
    CsrMatrix &operator=(CsrMatrix &&other) noexcept;
    void swap(CsrMatrix &other) noexcept;
    std::size_t rows() const noexcept { return m_rows; }
    std::size_t cols() const noexcept { return m_cols; }
    std::size_t nnz() const noexcept { return m_values.size(); }
    const std::vector<std::size_t> &indptr() const noexcept { return m_indptr; }
    const std::vector<std::size_t> &indices() const noexcept { return m_indices; }
    const std::vector<double> &values() const noexcept { return m_values; }

    private:
    std::size_t m_rows;
    std::size_t m_cols;
    std::vector<std::size_t> m_indptr;
    std::vector<std::size_t> m_indices;
    std::vector<double> m_values;
};

namespace detail
{
bool overlaps(const CsrMatrix &A, ConstVectorView x);
} // namespace detail

} // namespace cg
