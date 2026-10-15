#include <cg/backend/cuda/backend.hpp>
#include <cg/matrix/cuda/validation.hpp>
#include <cg/preconditioner/cuda/detail/kernels.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>

namespace cg
{
namespace cuda
{

JacobiPreconditioner::JacobiPreconditioner(DenseMatrixView A, const Context &context) : m_inverse(A.rows, context)
{
    detail::validate(A, context);
    if (A.rows == 0 || A.rows != A.cols)
    {
        throw std::invalid_argument("CUDA Jacobi requires a nonempty square matrix");
    }
    detail::initialize_jacobi(A, m_inverse.view(), context);
}
JacobiPreconditioner::JacobiPreconditioner(const CsrMatrix &A, const Context &context) : m_inverse(A.rows(), context)
{
    if (A.device() != context.device() || A.rows() == 0 || A.rows() != A.cols())
    {
        throw std::invalid_argument("CUDA Jacobi requires a square matrix on the selected device");
    }
    detail::initialize_jacobi(A.view(), m_inverse.view(), context);
}
double JacobiPreconditioner::apply_dot(ConstVectorView r, VectorView z, const Backend &backend) const
{
    return backend.multiply_dot(m_inverse.view(), r, z);
}

} // namespace cuda
} // namespace cg
