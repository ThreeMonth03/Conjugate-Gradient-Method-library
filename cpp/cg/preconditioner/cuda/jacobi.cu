#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/matrix/cuda/detail/validation.hpp>
#include <cg/preconditioner/cuda/detail/kernels.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <limits>
#include <utility>

namespace cg
{
namespace cuda
{

namespace
{

std::size_t storage_bytes(std::size_t size)
{
    // cudaMalloc aligns the base for both types. Every double-sized offset is
    // also int-aligned, so the flag needs no padding after the inverse span.
    static_assert(alignof(double) % alignof(int) == 0);
    if (size == 0)
    {
        return 0;
    }
    const auto inverse_bytes = cg::detail::checked_product(size, sizeof(double));
    if (inverse_bytes > std::numeric_limits<std::size_t>::max() - sizeof(int))
    {
        throw std::invalid_argument("CUDA Jacobi storage size overflows size_t");
    }
    return inverse_bytes + sizeof(int);
}

} // namespace

JacobiPreconditioner::JacobiPreconditioner(std::size_t size, const Context &context)
    : m_storage(detail::device_allocation<std::byte>(storage_bytes(size), context)),
      m_inverse(reinterpret_cast<double *>(m_storage.get()), size, context.device())
{
}
int *JacobiPreconditioner::invalid_flag() noexcept
{
    return reinterpret_cast<int *>(m_storage.get() + m_inverse.size() * sizeof(double));
}
JacobiPreconditioner::JacobiPreconditioner(DenseMatrixView A, const Context &context)
    : JacobiPreconditioner(A.rows(), context)
{
    detail::validate(A, context);
    if (A.rows() == 0 || A.rows() != A.cols())
    {
        throw std::invalid_argument("CUDA Jacobi requires a nonempty square matrix");
    }
    detail::initialize_jacobi(A, m_inverse, invalid_flag(), context);
}
JacobiPreconditioner::JacobiPreconditioner(const CsrMatrix &A, const Context &context)
    : JacobiPreconditioner(A.rows(), context)
{
    if (A.device() != context.device() || A.rows() == 0 || A.rows() != A.cols())
    {
        throw std::invalid_argument("CUDA Jacobi requires a square matrix on the selected device");
    }
    detail::initialize_jacobi(A.view(), m_inverse, invalid_flag(), context);
}
JacobiPreconditioner::JacobiPreconditioner(JacobiPreconditioner &&other) noexcept
    : m_storage(std::move(other.m_storage)), m_inverse(std::exchange(other.m_inverse, VectorView{}))
{
}
JacobiPreconditioner &JacobiPreconditioner::operator=(JacobiPreconditioner &&other) noexcept
{
    if (this != &other)
    {
        JacobiPreconditioner temporary(std::move(other));
        m_storage.swap(temporary.m_storage);
        std::swap(m_inverse, temporary.m_inverse);
    }
    return *this;
}
double JacobiPreconditioner::apply_and_dot(ConstVectorView r, VectorView z, const detail::Backend &backend) const
{
    return backend.multiply_and_dot(m_inverse, r, z);
}

} // namespace cuda
} // namespace cg
