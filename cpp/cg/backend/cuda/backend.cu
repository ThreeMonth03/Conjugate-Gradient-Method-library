#include <cg/backend/cuda/backend.hpp>
#include <cg/backend/cuda/detail/launch.hpp>
#include <cg/matrix/cuda/validation.hpp>
#include <cg/preconditioner/cuda/detail/kernels.hpp>
#include <cg/solver/cuda/detail/workspace.hpp>

#include <cmath>

namespace cg
{
namespace cuda
{
namespace
{

__device__ double block_sum(double value)
{
    __shared__ double sums[detail::block_size];
    sums[threadIdx.x] = value;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            sums[threadIdx.x] += sums[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const double result = sums[0];
    __syncthreads();
    return result;
}

__global__ void fill_kernel(VectorView output, double value)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        output.data[i] = value;
    }
}
__global__ void linear_kernel(ConstVectorView x, ConstVectorView y, VectorView output, double alpha, double beta)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        output.data[i] = alpha * x.data[i] + beta * y.data[i];
    }
}
__global__ void dot_kernel(ConstVectorView x, ConstVectorView y, double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        sum += x.data[i] * y.data[i];
    }
    const double total = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}
__global__ void finish_kernel(const double *partials, std::size_t count, double *scalar)
{
    const double total = block_sum(threadIdx.x < count ? partials[threadIdx.x] : 0.0);
    if (threadIdx.x == 0)
    {
        *scalar = total;
    }
}
__global__ void multiply_kernel(ConstVectorView factors, ConstVectorView r, VectorView z, double *partials, bool reduce)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double value = factors.data[i] * r.data[i];
        z.data[i] = value;
        if (reduce)
        {
            sum += r.data[i] * value;
        }
    }
    const double total = block_sum(sum);
    if (threadIdx.x == 0 && reduce)
    {
        partials[blockIdx.x] = total;
    }
}
__global__ void update_kernel(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap, double alpha,
                              double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        x.data[i] += alpha * p.data[i];
        const double value = r.data[i] - alpha * ap.data[i];
        r.data[i] = value;
        sum += value * value;
    }
    const double total = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}
__global__ void residual_kernel(ConstVectorView b, VectorView r, bool b_minus_ax, double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r.size;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double value = b_minus_ax ? b.data[i] - r.data[i] : r.data[i] - b.data[i];
        r.data[i] = value;
        sum += value * value;
    }
    const double total = block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}

// One cooperative block per row, with grid-stride rows for very large systems.
// mode: 0 product, 1 product + p^T Ap, 2 true residual + squared norm.
__global__ void dense_kernel(DenseMatrixView A, ConstVectorView x, ConstVectorView b, VectorView output, int mode,
                             bool b_minus_ax, double *partials)
{
    double metric = 0.0;
    for (std::size_t row = blockIdx.x; row < A.rows; row += gridDim.x)
    {
        double sum = 0.0;
        for (std::size_t column = threadIdx.x; column < A.cols; column += blockDim.x)
        {
            sum += A.data[row * A.row_stride + column] * x.data[column];
        }
        const double product = block_sum(sum);
        if (threadIdx.x == 0)
        {
            const double value = mode == 2 ? (b_minus_ax ? b.data[row] - product : product - b.data[row]) : product;
            output.data[row] = value;
            if (mode == 1)
            {
                metric += x.data[row] * value;
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

__global__ void csr_kernel(CsrMatrixView A, ConstVectorView x, ConstVectorView b, VectorView output, int mode,
                           bool b_minus_ax, double *partials)
{
    const auto *indptr = static_cast<const std::int64_t *>(A.indptr);
    const auto *indices = static_cast<const std::int64_t *>(A.indices);
    double metric = 0.0;
    for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < A.rows;
         row += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        double product = 0.0;
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            product += A.data[entry] * x.data[indices[entry]];
        }
        const double value = mode == 2 ? (b_minus_ax ? b.data[row] - product : product - b.data[row]) : product;
        output.data[row] = value;
        if (mode == 1)
        {
            metric += x.data[row] * value;
        }
        else if (mode == 2)
        {
            metric += value * value;
        }
    }
    const double total = block_sum(metric);
    if (threadIdx.x == 0 && mode != 0)
    {
        partials[blockIdx.x] = total;
    }
}

__global__ void stable_norm_kernel(ConstVectorView x, double *result)
{
    __shared__ double scales[detail::block_size], sums[detail::block_size];
    double scale = 0.0, sum = 1.0;
    for (std::size_t i = threadIdx.x; i < x.size; i += blockDim.x)
    {
        const double magnitude = fabs(x.data[i]);
        if (!isfinite(magnitude))
        {
            scale = magnitude;
            break;
        }
        if (magnitude != 0.0)
        {
            if (scale < magnitude)
            {
                const double ratio = scale / magnitude;
                sum = 1.0 + sum * ratio * ratio;
                scale = magnitude;
            }
            else
            {
                const double ratio = magnitude / scale;
                sum += ratio * ratio;
            }
        }
    }
    scales[threadIdx.x] = scale;
    sums[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2)
    {
        if (threadIdx.x < stride)
        {
            const double other = scales[threadIdx.x + stride];
            const double current = scales[threadIdx.x];
            if (isnan(current) || isnan(other))
            {
                scales[threadIdx.x] = nan("");
            }
            else if (isinf(current) || isinf(other))
            {
                scales[threadIdx.x] = INFINITY;
            }
            else if (current < other)
            {
                const double ratio = current / other;
                sums[threadIdx.x] = sums[threadIdx.x + stride] + sums[threadIdx.x] * ratio * ratio;
                scales[threadIdx.x] = other;
            }
            else if (current != 0.0)
            {
                const double ratio = other / current;
                sums[threadIdx.x] += sums[threadIdx.x + stride] * ratio * ratio;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *result = scales[0] * sqrt(sums[0]);
    }
}

__global__ void dense_jacobi_kernel(DenseMatrixView A, VectorView inverse, int *invalid)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < A.rows;
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double diagonal = A.data[i * A.row_stride + i];
        const double value = 1.0 / diagonal;
        if (!isfinite(diagonal) || diagonal <= 0.0 || !isfinite(value) || value <= 0.0)
        {
            atomicExch(invalid, 1);
        }
        inverse.data[i] = value;
    }
}
__global__ void csr_jacobi_kernel(CsrMatrixView A, VectorView inverse, int *invalid)
{
    const auto *indptr = static_cast<const std::int64_t *>(A.indptr);
    const auto *indices = static_cast<const std::int64_t *>(A.indices);
    for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < A.rows;
         row += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        double diagonal = 0.0;
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            if (indices[entry] == static_cast<std::int64_t>(row))
            {
                diagonal = A.data[entry];
                break;
            }
        }
        const double value = 1.0 / diagonal;
        if (!isfinite(diagonal) || diagonal <= 0.0 || !isfinite(value) || value <= 0.0)
        {
            atomicExch(invalid, 1);
        }
        inverse.data[row] = value;
    }
}

void launch_check()
{
    detail::check(cudaGetLastError(), "CUDA kernel launch");
}
void same_size(ConstVectorView a, ConstVectorView b, const Context &context)
{
    detail::validate(a, context, false);
    detail::validate(b, context, false);
    if (a.size != b.size)
    {
        throw std::invalid_argument("CUDA vector dimensions do not match");
    }
}
void no_overlap(ConstVectorView a, ConstVectorView b)
{
    if (detail::overlaps(a, b))
    {
        throw std::invalid_argument("CUDA operation requires disjoint storage");
    }
}
void exact_or_disjoint(ConstVectorView a, ConstVectorView b)
{
    if (a.data != b.data)
    {
        no_overlap(a, b);
    }
}
void validate_apply(DenseMatrixView A, ConstVectorView x, VectorView output, const Context &context)
{
    detail::validate(A, context, false);
    detail::validate(x, context, false);
    detail::validate(output, context, false);
    if (A.cols != x.size || A.rows != output.size)
    {
        throw std::invalid_argument("CUDA matrix-vector dimensions do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("CUDA matrix and output storage overlap");
    }
}
void validate_apply(const CsrMatrix &A, ConstVectorView x, VectorView output, const Context &context)
{
    detail::validate(x, context, false);
    detail::validate(output, context, false);
    if (A.device() != context.device() || A.cols() != x.size || A.rows() != output.size)
    {
        throw std::invalid_argument("CUDA CSR dimensions or device do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("CUDA CSR and output storage overlap");
    }
}
unsigned int dense_blocks(std::size_t rows)
{
    return static_cast<unsigned int>(std::min<std::size_t>(std::max<std::size_t>(rows, 1), detail::max_blocks));
}

} // namespace

namespace detail
{
void fill(VectorView output, double value, const Context &context)
{
    DeviceGuard guard(context.device());
    fill_kernel<<<blocks(output.size), block_size, 0, stream(context)>>>(output, value);
    launch_check();
}
template <class Launch> void prepare_jacobi(const Context &context, Launch launch)
{
    DeviceGuard guard(context.device());
    DeviceAllocation<int> invalid(1, context);
    check(cudaMemsetAsync(invalid.data(), 0, sizeof(int), stream(context)), "cudaMemsetAsync Jacobi");
    launch(invalid.data());
    launch_check();
    int host = 0;
    check(cudaMemcpyAsync(&host, invalid.data(), sizeof(int), cudaMemcpyDeviceToHost, stream(context)),
          "cudaMemcpyAsync Jacobi validation");
    context.synchronize();
    if (host != 0)
    {
        throw std::invalid_argument("CUDA Jacobi requires a finite positive diagonal and inverse");
    }
}
void initialize_jacobi(DenseMatrixView A, VectorView inverse, const Context &context)
{
    prepare_jacobi(context, [&](int *invalid)
                   { dense_jacobi_kernel<<<blocks(A.rows), block_size, 0, stream(context)>>>(A, inverse, invalid); });
}
void initialize_jacobi(CsrMatrixView A, VectorView inverse, const Context &context)
{
    prepare_jacobi(context, [&](int *invalid)
                   { csr_jacobi_kernel<<<blocks(A.rows), block_size, 0, stream(context)>>>(A, inverse, invalid); });
}
} // namespace detail

Backend::Backend(const Context &context, Workspace &workspace, bool fused, DenseKernel kernel)
    : m_context(context), m_workspace(workspace), m_fused(fused), m_kernel(kernel)
{
    if (workspace.device() != context.device() || workspace.stream() != context.stream())
    {
        throw std::invalid_argument("CUDA workspace must match the device and stream");
    }
    if (kernel != DenseKernel::native && kernel != DenseKernel::cublas)
    {
        throw std::invalid_argument("unsupported CUDA dense kernel");
    }
}
double Backend::finish_reduction(std::size_t blocks) const
{
    auto &work = *m_workspace.m_impl;
    finish_kernel<<<1, detail::block_size, 0, detail::stream(m_context)>>>(work.partials.data(), blocks,
                                                                           work.scalar.data());
    launch_check();
    detail::check(cudaMemcpyAsync(work.host.data, work.scalar.data(), sizeof(double), cudaMemcpyDeviceToHost,
                                  detail::stream(m_context)),
                  "cudaMemcpyAsync reduction");
    m_context.synchronize();
    return *work.host.data;
}
void Backend::copy(ConstVectorView source, VectorView destination) const
{
    same_size(source, destination, m_context);
    exact_or_disjoint(source, destination);
    detail::DeviceGuard guard(m_context.device());
    if (source.size != 0 && source.data != destination.data)
    {
        detail::check(cudaMemcpyAsync(destination.data, source.data, source.size * sizeof(double),
                                      cudaMemcpyDeviceToDevice, detail::stream(m_context)),
                      "cudaMemcpyAsync copy");
    }
}
double Backend::dot(ConstVectorView x, ConstVectorView y) const
{
    same_size(x, y, m_context);
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = detail::blocks(x.size);
    dot_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(x, y, m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
double Backend::norm(ConstVectorView x) const
{
    return norm_from_squared(x, dot(x, x));
}
double Backend::norm_from_squared(ConstVectorView x, double squared) const
{
    if (std::isfinite(squared) && squared >= std::numeric_limits<double>::min())
    {
        return std::sqrt(squared);
    }
    detail::validate(x, m_context, false);
    detail::DeviceGuard guard(m_context.device());
    auto &work = *m_workspace.m_impl;
    stable_norm_kernel<<<1, detail::block_size, 0, detail::stream(m_context)>>>(x, work.scalar.data());
    launch_check();
    detail::check(cudaMemcpyAsync(work.host.data, work.scalar.data(), sizeof(double), cudaMemcpyDeviceToHost,
                                  detail::stream(m_context)),
                  "cudaMemcpyAsync stable norm");
    m_context.synchronize();
    return *work.host.data;
}
double Backend::multiply_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const
{
    same_size(factors, r, m_context);
    same_size(r, z, m_context);
    no_overlap(factors, z);
    no_overlap(r, z);
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = detail::blocks(r.size);
    multiply_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(
        factors, r, z, m_workspace.m_impl->partials.data(), m_fused);
    launch_check();
    return m_fused ? finish_reduction(blocks) : dot(r, z);
}
void Backend::axpy(double alpha, ConstVectorView x, VectorView y) const
{
    same_size(x, y, m_context);
    exact_or_disjoint(x, y);
    detail::DeviceGuard guard(m_context.device());
    linear_kernel<<<detail::blocks(x.size), detail::block_size, 0, detail::stream(m_context)>>>(x, y, y, alpha, 1.0);
    launch_check();
}
void Backend::axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const
{
    same_size(x, y, m_context);
    same_size(x, output, m_context);
    exact_or_disjoint(x, output);
    exact_or_disjoint(y, output);
    detail::DeviceGuard guard(m_context.device());
    linear_kernel<<<detail::blocks(x.size), detail::block_size, 0, detail::stream(m_context)>>>(x, y, output, alpha,
                                                                                                beta);
    launch_check();
}
void Backend::apply(DenseMatrixView A, ConstVectorView x, VectorView output) const
{
    validate_apply(A, x, output, m_context);
    detail::DeviceGuard guard(m_context.device());
    if (m_kernel == DenseKernel::cublas)
    {
        const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
        if (A.rows > limit || A.cols > limit || A.row_stride > limit)
        {
            throw std::invalid_argument("CUDA cuBLAS dimensions or row stride exceed int32 range");
        }
        if (A.rows == 0)
        {
            return;
        }
        if (A.cols == 0)
        {
            detail::fill(output, 0.0, m_context);
            return;
        }
        const double alpha = 1.0, beta = 0.0;
        detail::check(cublasDgemv(m_workspace.m_impl->handle.value, CUBLAS_OP_T, static_cast<int>(A.cols),
                                  static_cast<int>(A.rows), &alpha, A.data, static_cast<int>(A.row_stride), x.data, 1,
                                  &beta, output.data, 1),
                      "cublasDgemv");
    }
    else
    {
        dense_kernel<<<dense_blocks(A.rows), detail::block_size, 0, detail::stream(m_context)>>>(A, x, {}, output, 0,
                                                                                                 true, nullptr);
        launch_check();
    }
}
double Backend::apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const
{
    same_size(p, ap, m_context);
    validate_apply(A, p, ap, m_context);
    if (!m_fused || m_kernel == DenseKernel::cublas)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = dense_blocks(A.rows);
    dense_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(A, p, {}, ap, 1, true,
                                                                               m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
double Backend::residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r, bool b_minus_ax) const
{
    same_size(b, r, m_context);
    no_overlap(b, r);
    validate_apply(A, x, r, m_context);
    detail::DeviceGuard guard(m_context.device());
    if (m_fused && m_kernel == DenseKernel::native)
    {
        const auto blocks = dense_blocks(A.rows);
        dense_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(A, x, b, r, 2, b_minus_ax,
                                                                                   m_workspace.m_impl->partials.data());
        launch_check();
        return finish_reduction(blocks);
    }
    apply(A, x, r);
    if (!m_fused)
    {
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return dot(r, r);
    }
    const auto blocks = detail::blocks(r.size);
    residual_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(b, r, b_minus_ax,
                                                                                  m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
void Backend::apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const
{
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    validate_apply(A, x, output, m_context);
    detail::DeviceGuard guard(m_context.device());
    csr_kernel<<<detail::blocks(A.rows()), detail::block_size, 0, detail::stream(m_context)>>>(A.view(), x, {}, output,
                                                                                               0, true, nullptr);
    launch_check();
}
double Backend::apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const
{
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    same_size(p, ap, m_context);
    validate_apply(A, p, ap, m_context);
    if (!m_fused)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = detail::blocks(A.rows());
    csr_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(A.view(), p, {}, ap, 1, true,
                                                                             m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
double Backend::residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r, bool b_minus_ax) const
{
    if (m_kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    same_size(b, r, m_context);
    no_overlap(b, r);
    validate_apply(A, x, r, m_context);
    if (!m_fused)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return dot(r, r);
    }
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = detail::blocks(A.rows());
    csr_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(A.view(), x, b, r, 2, b_minus_ax,
                                                                             m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
double Backend::update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap, double alpha) const
{
    same_size(x, r, m_context);
    same_size(x, p, m_context);
    same_size(x, ap, m_context);
    no_overlap(x, r);
    no_overlap(x, p);
    no_overlap(x, ap);
    no_overlap(r, p);
    no_overlap(r, ap);
    if (!m_fused)
    {
        axpy(alpha, p, x);
        axpy(-alpha, ap, r);
        return dot(r, r);
    }
    detail::DeviceGuard guard(m_context.device());
    const auto blocks = detail::blocks(x.size);
    update_kernel<<<blocks, detail::block_size, 0, detail::stream(m_context)>>>(x, r, p, ap, alpha,
                                                                                m_workspace.m_impl->partials.data());
    launch_check();
    return finish_reduction(blocks);
}
void Backend::update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale) const
{
    no_overlap(r, p);
    axpby(residual_scale, r, beta, p, p);
}

} // namespace cuda
} // namespace cg
