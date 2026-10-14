#ifndef CG_CSR_HPP
#define CG_CSR_HPP

#include "cg/storage.hpp"

#include <cstddef>
#include <vector>

namespace cg
{

// Canonical owning CSR: row pointers are monotone and column indices within
// each row are strictly increasing. Construction rejects duplicates and
// unsorted columns rather than silently changing the supplied matrix.
class CsrMatrix
{
  public:
    CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> indptr,
              std::vector<std::size_t> indices, std::vector<double> values);
    CsrMatrix(const CsrMatrix &) = default;
    CsrMatrix &operator=(const CsrMatrix &) = default;
    CsrMatrix(CsrMatrix &&other) noexcept;
    CsrMatrix &operator=(CsrMatrix &&other) noexcept;
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    std::size_t nnz() const noexcept { return values_.size(); }
    const std::vector<std::size_t> &indptr() const noexcept { return indptr_; }
    const std::vector<std::size_t> &indices() const noexcept { return indices_; }
    const std::vector<double> &values() const noexcept { return values_; }

  private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<std::size_t> indptr_;
    std::vector<std::size_t> indices_;
    std::vector<double> values_;
};

namespace detail
{
bool overlaps(const CsrMatrix &A, ConstVectorView x);
} // namespace detail

} // namespace cg

#endif
