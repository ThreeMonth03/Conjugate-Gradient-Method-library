#pragma once

#include <cg/backend/cuda/context.hpp>
#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/solver/cuda/workspace.hpp>

namespace cg
{
namespace cuda
{

enum class DenseKernel
{
    native,
    cublas
};

// Execution adapter for the shared solver engine. Public solve_into validates
// allocation/device attributes once; its kernels reuse the validated views.
class Backend
{
    public:
    Backend(const Context &context, Workspace &workspace, bool fused = true, DenseKernel kernel = DenseKernel::native);
    void copy(ConstVectorView source, VectorView destination) const;
    double dot(ConstVectorView x, ConstVectorView y) const;
    double norm(ConstVectorView x) const;
    double norm_from_squared(ConstVectorView x, double squared) const;
    double multiply_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    double residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    double residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    double update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap, double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale = 1.0) const;

    private:
    double finish_reduction(std::size_t blocks) const;
    Context m_context;
    Workspace &m_workspace;
    bool m_fused;
    DenseKernel m_kernel;
};

} // namespace cuda
} // namespace cg
