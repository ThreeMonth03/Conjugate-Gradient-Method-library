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
    // Asynchronous scalar destinations are borrowed device storage, disjoint
    // from vector inputs and internal partials, and live until stream completion.
    void dot_into(ConstVectorView x, ConstVectorView y, double *result) const;
    double norm(ConstVectorView x) const;
    double norm_from_squared(ConstVectorView x, double squared) const;
    // squared and result must be distinct scalar slots.
    void norm_from_squared_into(ConstVectorView x, const double *squared, double *result) const;
    double multiply_and_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void multiply_and_dot_into(ConstVectorView factors, ConstVectorView r, VectorView z, double *result) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_and_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    void apply_and_dot_into(DenseMatrixView A, ConstVectorView p, VectorView ap, double *result) const;
    double residual_squared_norm(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                                 ResidualSign sign = ResidualSign::rhs_minus_product) const;
    void residual_squared_norm_into(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                                    double *result, ResidualSign sign = ResidualSign::rhs_minus_product) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_and_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    void apply_and_dot_into(const CsrMatrix &A, ConstVectorView p, VectorView ap, double *result) const;
    double residual_squared_norm(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                                 ResidualSign sign = ResidualSign::rhs_minus_product) const;
    void residual_squared_norm_into(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                                    double *result, ResidualSign sign = ResidualSign::rhs_minus_product) const;
    void finish_reduction_into(std::size_t blocks, double *result) const;
    double update_x_r_squared_norm(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                                   double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale = 1.0) const;

    private:
    double finish_reduction(std::size_t blocks) const;
    double read_scalar(const double *result) const;
    ExecutionResources &m_resources;
    bool m_fused;
    DenseKernel m_kernel;
};

} // namespace detail
} // namespace cuda
} // namespace cg
