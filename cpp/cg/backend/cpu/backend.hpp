#pragma once

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
    // Stable L2 norm. norm2 is the squared sum used by the recurrence and may
    // overflow or underflow when the norm itself is still representable.
    double norm(ConstVectorView x) const;
    double norm2(ConstVectorView x) const;
    void multiply(ConstVectorView x, ConstVectorView y, VectorView output) const;
    // z = factors * r; return dot(r,z). The output must be disjoint from inputs.
    double multiply_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    // The residual sign is explicit to define the explicit-residual formulation.
    double residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    double residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    double update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap, double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale = 1.0) const;

    private:
    int m_threads;
    bool m_fused;
    DenseKernel m_dense_kernel;
};

namespace detail
{
// The view was validated by the operation that produced squared_norm.
double norm_from_squared(ConstVectorView x, double squared_norm);
} // namespace detail

} // namespace cg
