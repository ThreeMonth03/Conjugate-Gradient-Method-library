#pragma once

#include <cg/runtime/cuda/detail/launch.hpp>
#include <cg/storage/cuda/vector_view.hpp>

#include <cmath>
#include <limits>

namespace cg
{
namespace cuda
{
namespace detail
{

__device__ inline double block_sum(double value)
{
    __shared__ double sums[block_size];
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

// Call uniformly from one complete block of block_size threads.
// The scalar fast path and fallback retain the backend norm's arithmetic tree.
constexpr double minimum_norm_squared = std::numeric_limits<double>::min();

__device__ inline void block_stable_norm_into(ConstVectorView x, const double *squared, double *result)
{
    if (squared != nullptr && isfinite(*squared) && *squared >= minimum_norm_squared)
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

} // namespace detail
} // namespace cuda
} // namespace cg
