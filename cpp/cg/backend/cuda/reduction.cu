#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/backend/cuda/detail/reduction.cuh>
#include <cg/backend/cuda/detail/validation.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>
#include <cfloat>
#include <cmath>
#include <limits>

namespace cg
{
namespace cuda
{
namespace detail
{
namespace
{

__global__ void dot_kernel(ConstVectorView x, ConstVectorView y, double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        sum += x.data()[i] * y.data()[i];
    }
    const double total = detail::block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}

__global__ void finish_kernel(const double *partials, std::size_t count, double *scalar)
{
    const double total = detail::block_sum(threadIdx.x < count ? partials[threadIdx.x] : 0.0);
    if (threadIdx.x == 0)
    {
        *scalar = total;
    }
}

__global__ void stable_norm_kernel(ConstVectorView x, const double *squared, double *result)
{
    if (squared != nullptr && isfinite(*squared) && *squared >= DBL_MIN)
    {
        if (threadIdx.x == 0)
        {
            *result = sqrt(*squared);
        }
        return;
    }
    __shared__ double scales[detail::block_size], sums[detail::block_size];
    double scale = 0.0, sum = 1.0;
    for (std::size_t i = threadIdx.x; i < x.size(); i += blockDim.x)
    {
        const double magnitude = fabs(x.data()[i]);
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

} // namespace

void Backend::finish_reduction_into(std::size_t blocks, double *result) const
{
    detail::DeviceGuard guard(m_resources.context.device());
    finish_kernel<<<1, detail::block_size, 0, detail::stream(m_resources.context)>>>(m_resources.partials, blocks,
                                                                                     result);
    detail::launch_check();
}

double Backend::read_scalar(const double *result) const
{
    detail::DeviceGuard guard(m_resources.context.device());
    auto &work = m_resources;
    detail::check(
        cudaMemcpyAsync(&work.host, result, sizeof(double), cudaMemcpyDeviceToHost, detail::stream(work.context)),
        "cudaMemcpyAsync reduction");
    work.context.synchronize();
    return work.host;
}

double Backend::finish_reduction(std::size_t blocks) const
{
    finish_reduction_into(blocks, m_resources.scalar);
    return read_scalar(m_resources.scalar);
}

void Backend::dot_into(ConstVectorView x, ConstVectorView y, double *result) const
{
    detail::same_size(x, y, m_resources.context);
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::blocks(x.size());
    dot_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(x, y, m_resources.partials);
    detail::launch_check();
    finish_reduction_into(blocks, result);
}

double Backend::dot(ConstVectorView x, ConstVectorView y) const
{
    dot_into(x, y, m_resources.scalar);
    return read_scalar(m_resources.scalar);
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
    detail::validate(x, m_resources.context, false);
    detail::DeviceGuard guard(m_resources.context.device());
    auto &work = m_resources;
    stable_norm_kernel<<<1, detail::block_size, 0, detail::stream(m_resources.context)>>>(x, nullptr, work.scalar);
    detail::launch_check();
    detail::check(cudaMemcpyAsync(&work.host, work.scalar, sizeof(double), cudaMemcpyDeviceToHost,
                                  detail::stream(m_resources.context)),
                  "cudaMemcpyAsync stable norm");
    m_resources.context.synchronize();
    return work.host;
}

void Backend::norm_from_squared_into(ConstVectorView x, const double *squared, double *result) const
{
    detail::validate(x, m_resources.context, false);
    detail::DeviceGuard guard(m_resources.context.device());
    stable_norm_kernel<<<1, detail::block_size, 0, detail::stream(m_resources.context)>>>(x, squared, result);
    detail::launch_check();
}

} // namespace detail
} // namespace cuda
} // namespace cg
