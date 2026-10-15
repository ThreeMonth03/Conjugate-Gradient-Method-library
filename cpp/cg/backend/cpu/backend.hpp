#pragma once

#include <cg/backend/residual_sign.hpp>
#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/storage/vector_view.hpp>

namespace cg
{

class CsrMatrix;

enum class DenseKernel
{
    native,
    blas
};

// All methods are synchronous. Thread settings are local to this context and
// never change OpenMP's process-wide defaults. Inputs must remain alive and
// unmodified by other threads until the operation returns.
// The optional BLAS kernel uses a private sequential provider and only accepts
// threads=1. It replaces dense matvecs; vector operations remain native. CSR
// operations reject this kernel selection.
class CpuBackend
{
    public:
    explicit CpuBackend(int threads = 1, bool fused = true, DenseKernel dense_kernel = DenseKernel::native);
    int threads() const noexcept { return m_threads; }
    bool fused() const noexcept { return m_fused; }
    DenseKernel dense_kernel() const noexcept { return m_dense_kernel; }
    static bool openmp_enabled() noexcept;
    static bool blas_enabled() noexcept;
    static const char *blas_info() noexcept;

    void copy(ConstVectorView source, VectorView destination) const;
    double dot(ConstVectorView x, ConstVectorView y) const;
    // Stable L2 norm. squared_norm is the squared sum used by the recurrence and may
    // overflow or underflow when the norm itself is still representable.
    double norm(ConstVectorView x) const;
    double squared_norm(ConstVectorView x) const;
    void multiply(ConstVectorView x, ConstVectorView y, VectorView output) const;
    // z = factors * r; return dot(r,z). The output must be disjoint from inputs.
    double multiply_and_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_and_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    // The residual sign is explicit to define the explicit-residual formulation.
    double residual_squared_norm(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                                 ResidualSign sign = ResidualSign::rhs_minus_product) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_and_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    double residual_squared_norm(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                                 ResidualSign sign = ResidualSign::rhs_minus_product) const;
    double update_x_r_squared_norm(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                                   double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale = 1.0) const;

    private:
    int m_threads;
    bool m_fused;
    DenseKernel m_dense_kernel;
};

} // namespace cg
