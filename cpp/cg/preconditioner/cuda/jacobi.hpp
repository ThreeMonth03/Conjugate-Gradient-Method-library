#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/storage/cuda/device_memory.hpp>
#include <cg/storage/cuda/vector_view.hpp>

namespace cg
{
namespace cuda
{

namespace detail
{
class Backend;
}

class JacobiPreconditioner
{
    public:
    JacobiPreconditioner(DenseMatrixView A, const Context &context);
    JacobiPreconditioner(const CsrMatrix &A, const Context &context);
    JacobiPreconditioner(const JacobiPreconditioner &) = delete;
    JacobiPreconditioner &operator=(const JacobiPreconditioner &) = delete;
    JacobiPreconditioner(JacobiPreconditioner &&other) noexcept;
    JacobiPreconditioner &operator=(JacobiPreconditioner &&other) noexcept;
    std::size_t size() const noexcept { return m_inverse.size(); }
    int device() const noexcept { return m_inverse.device(); }
    ConstVectorView inverse_diagonal() const & noexcept { return m_inverse; }
    ConstVectorView inverse_diagonal() const && = delete;
    double apply_and_dot(ConstVectorView r, VectorView z, const detail::Backend &backend) const;

    private:
    JacobiPreconditioner(std::size_t size, const Context &context);
    int *invalid_flag() noexcept;
    // The inverse and its setup-only validation flag share one owning buffer.
    // The public span covers only the inverse, preserving its size/byte meaning.
    detail::DeviceAllocation<std::byte> m_storage;
    VectorView m_inverse;
};

} // namespace cuda
} // namespace cg
