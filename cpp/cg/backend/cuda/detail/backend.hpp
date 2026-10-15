#pragma once

#include <cg/backend/cuda/dense_kernel.hpp>
#include <cg/backend/residual_sign.hpp>
#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

struct ExecutionResources;

// Execution adapter for the shared solver engine. Public solve_into validates
// allocation/device attributes once; its kernels reuse the validated views.
class Backend
{
    public:
    Backend(ExecutionResources &resources, bool fused = true, DenseKernel kernel = DenseKernel::native);
    void copy(ConstVectorView source, VectorView destination) const;
    double dot(ConstVectorView x, ConstVectorView y) const;
    double norm(ConstVectorView x) const;
    double norm_from_squared(ConstVectorView x, double squared) const;
    double multiply_and_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_and_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
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
    double finish_reduction(std::size_t blocks) const;
    ExecutionResources &m_resources;
    bool m_fused;
    DenseKernel m_kernel;
};

} // namespace detail
} // namespace cuda
} // namespace cg
