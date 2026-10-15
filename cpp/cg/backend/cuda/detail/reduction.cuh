#pragma once

#include <cg/backend/cuda/detail/launch.hpp>

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

} // namespace detail
} // namespace cuda
} // namespace cg
