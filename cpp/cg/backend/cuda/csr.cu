#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/backend/cuda/detail/reduction.cuh>
#include <cg/backend/cuda/detail/validation.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{
namespace
{

__global__ void csr_kernel(CsrMatrixView A, ConstVectorView x, ConstVectorView b, VectorView output, int mode,
                           bool b_minus_ax, double *partials)
{
    const auto *indptr = static_cast<const std::int64_t *>(A.indptr());
    const auto *indices = static_cast<const std::int64_t *>(A.indices());
    double metric = 0.0;
    for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < A.rows();
         row += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        double product = 0.0;
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            product += A.data()[entry] * x.data()[indices[entry]];
        }
        const double value = mode == 2 ? (b_minus_ax ? b.data()[row] - product : product - b.data()[row]) : product;
        output.data()[row] = value;
        if (mode == 1)
        {
            metric += x.data()[row] * value;
        }
        else if (mode == 2)
        {
            metric += value * value;
        }
    }
    const double total = detail::block_sum(metric);
    if (threadIdx.x == 0 && mode != 0)
    {
        partials[blockIdx.x] = total;
    }
}

} // namespace

void Backend::apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const
{
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    detail::validate_apply(A, x, output, m_resources.context);
    detail::DeviceGuard guard(m_resources.context.device());
    csr_kernel<<<detail::blocks(A.rows()), detail::block_size, 0, detail::stream(m_resources.context)>>>(
        A.view(), x, {}, output, 0, true, nullptr);
    detail::launch_check();
}

double Backend::apply_and_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const
{
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    detail::same_size(p, ap, m_resources.context);
    detail::validate_apply(A, p, ap, m_resources.context);
    if (!m_fused)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::blocks(A.rows());
    csr_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(A.view(), p, {}, ap, 1, true,
                                                                                       m_resources.partials);
    detail::launch_check();
    return finish_reduction(blocks);
}

double Backend::residual_squared_norm(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                                      ResidualSign sign) const
{
    const bool b_minus_ax = sign == ResidualSign::rhs_minus_product;
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    detail::same_size(b, r, m_resources.context);
    detail::no_overlap(b, r);
    detail::validate_apply(A, x, r, m_resources.context);
    if (!m_fused)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return dot(r, r);
    }
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::blocks(A.rows());
    csr_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(A.view(), x, b, r, 2, b_minus_ax,
                                                                                       m_resources.partials);
    detail::launch_check();
    return finish_reduction(blocks);
}

} // namespace detail
} // namespace cuda
} // namespace cg
