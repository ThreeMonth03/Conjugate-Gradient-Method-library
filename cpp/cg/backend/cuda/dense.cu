#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/backend/cuda/detail/reduction.cuh>
#include <cg/backend/cuda/detail/validation.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>
#include <cg/storage/cuda/detail/fill.hpp>
#include <limits>

namespace cg
{
namespace cuda
{
namespace detail
{
namespace
{

__global__ void dense_kernel(DenseMatrixView A, ConstVectorView x, ConstVectorView b, VectorView output, int mode,
                             bool b_minus_ax, double *partials)
{
    double metric = 0.0;
    for (std::size_t row = blockIdx.x; row < A.rows(); row += gridDim.x)
    {
        double sum = 0.0;
        for (std::size_t column = threadIdx.x; column < A.cols(); column += blockDim.x)
        {
            sum += A.data()[row * A.row_stride() + column] * x.data()[column];
        }
        const double product = detail::block_sum(sum);
        if (threadIdx.x == 0)
        {
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
    }
    if (threadIdx.x == 0 && mode != 0)
    {
        partials[blockIdx.x] = metric;
    }
}

__global__ void residual_kernel(ConstVectorView b, VectorView r, bool b_minus_ax, double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double value = b_minus_ax ? b.data()[i] - r.data()[i] : r.data()[i] - b.data()[i];
        r.data()[i] = value;
        sum += value * value;
    }
    const double total = detail::block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}

} // namespace

void Backend::apply(DenseMatrixView A, ConstVectorView x, VectorView output) const
{
    detail::validate_apply(A, x, output, m_resources.context);
    detail::DeviceGuard guard(m_resources.context.device());
    if (m_kernel == DenseKernel::cublas)
    {
        const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (A.rows() > limit || A.cols() > limit || A.row_stride() > limit)
        {
            throw std::invalid_argument("CUDA cuBLAS dimensions or row stride exceed int32 range");
        }
        if (A.rows() == 0)
        {
            return;
        }
        if (A.cols() == 0)
        {
            detail::fill(output, 0.0, m_resources.context);
            return;
        }
        const double alpha = 1.0, beta = 0.0;
        detail::check(cublasDgemv(m_resources.cublas->handle.value, CUBLAS_OP_T, static_cast<int>(A.cols()),
                                  static_cast<int>(A.rows()), &alpha, A.data(), static_cast<int>(A.row_stride()),
                                  x.data(), 1, &beta, output.data(), 1),
                      "cublasDgemv");
    }
    else
    {
        dense_kernel<<<detail::dense_blocks(A.rows()), detail::block_size, 0, detail::stream(m_resources.context)>>>(
            A, x, {}, output, 0, true, nullptr);
        detail::launch_check();
    }
}

double Backend::apply_and_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const
{
    detail::same_size(p, ap, m_resources.context);
    detail::validate_apply(A, p, ap, m_resources.context);
    if (!m_fused || m_kernel == DenseKernel::cublas)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::dense_blocks(A.rows());
    dense_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(A, p, {}, ap, 1, true,
                                                                                         m_resources.partials.get());
    detail::launch_check();
    return finish_reduction(blocks);
}

double Backend::residual_squared_norm(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                                      ResidualSign sign) const
{
    const bool b_minus_ax = sign == ResidualSign::rhs_minus_product;
    detail::same_size(b, r, m_resources.context);
    detail::no_overlap(b, r);
    detail::validate_apply(A, x, r, m_resources.context);
    detail::DeviceGuard guard(m_resources.context.device());
    if (m_fused && m_kernel == DenseKernel::native)
    {
        const auto blocks = detail::dense_blocks(A.rows());
        dense_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(
            A, x, b, r, 2, b_minus_ax, m_resources.partials.get());
        detail::launch_check();
        return finish_reduction(blocks);
    }
    apply(A, x, r);
    if (!m_fused)
    {
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return dot(r, r);
    }
    const auto blocks = detail::blocks(r.size());
    residual_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(b, r, b_minus_ax,
                                                                                            m_resources.partials.get());
    detail::launch_check();
    return finish_reduction(blocks);
}

} // namespace detail
} // namespace cuda
} // namespace cg
