#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/storage/cuda/vector.hpp>

namespace cg
{
namespace cuda
{

class Backend;

class JacobiPreconditioner
{
    public:
    JacobiPreconditioner(DenseMatrixView A, const Context &context);
    JacobiPreconditioner(const CsrMatrix &A, const Context &context);
    std::size_t size() const noexcept { return m_inverse.size(); }
    int device() const noexcept { return m_inverse.device(); }
    ConstVectorView inverse_diagonal() const noexcept { return m_inverse.view(); }
    double apply_dot(ConstVectorView r, VectorView z, const Backend &backend) const;

    private:
    Vector m_inverse;
};

} // namespace cuda
} // namespace cg
