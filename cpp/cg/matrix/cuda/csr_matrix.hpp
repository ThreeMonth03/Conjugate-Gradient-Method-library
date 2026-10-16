#pragma once

#include <cg/matrix/csr_matrix.hpp>
#include <cg/matrix/cuda/csr_matrix_view.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/device_memory.hpp>
#include <cg/storage/cuda/vector.hpp>

#include <cstdint>

namespace cg
{
namespace cuda
{

// Device snapshots canonicalize neither ordering nor duplicates: invalid CSR
// input is rejected after a one-time setup copy of its index arrays.
class CsrMatrix
{
    public:
    CsrMatrix(const cg::CsrMatrix &host, const Context &context);
    CsrMatrix(CsrMatrixView source, const Context &context);
    ~CsrMatrix() = default;
    CsrMatrix(const CsrMatrix &) = delete;
    CsrMatrix &operator=(const CsrMatrix &) = delete;
    CsrMatrix(CsrMatrix &&) noexcept;
    CsrMatrix &operator=(CsrMatrix &&) noexcept;
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t nnz() const noexcept;
    int device() const noexcept;
    CsrMatrixView view() const & noexcept;
    CsrMatrixView view() const && = delete;

    private:
    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    detail::DeviceAllocation<std::int64_t> m_indptr;
    detail::DeviceAllocation<std::int64_t> m_indices;
    Vector m_values;
};

} // namespace cuda
} // namespace cg
