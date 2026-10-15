#pragma once

#include <cg/backend/cuda/context.hpp>
#include <cg/matrix/csr_matrix.hpp>
#include <cg/matrix/cuda/csr_matrix_view.hpp>

#include <memory>

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
    ~CsrMatrix();
    CsrMatrix(const CsrMatrix &) = delete;
    CsrMatrix &operator=(const CsrMatrix &) = delete;
    CsrMatrix(CsrMatrix &&) noexcept;
    CsrMatrix &operator=(CsrMatrix &&) noexcept;
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t nnz() const noexcept;
    int device() const noexcept;
    CsrMatrixView view() const noexcept;

    private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace cuda
} // namespace cg
