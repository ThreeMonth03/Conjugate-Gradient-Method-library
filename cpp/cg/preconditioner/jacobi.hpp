#pragma once

#include <cg/backend/cpu/backend.hpp>
#include <cg/storage/vector.hpp>

namespace cg
{

// Immutable snapshot of an SPD matrix's inverse diagonal. Preparing it may
// allocate; applying it uses caller-owned vector storage and never allocates.
class JacobiPreconditioner
{
    public:
    explicit JacobiPreconditioner(DenseMatrixView A);
    explicit JacobiPreconditioner(const CsrMatrix &A);
    std::size_t size() const noexcept { return m_inverse.size(); }
    ConstVectorView inverse_diagonal() const noexcept { return m_inverse.view(); }
    void apply(ConstVectorView r, VectorView z, const CpuBackend &backend) const;
    double apply_dot(ConstVectorView r, VectorView z, const CpuBackend &backend) const;

    private:
    Vector m_inverse;
};

} // namespace cg
