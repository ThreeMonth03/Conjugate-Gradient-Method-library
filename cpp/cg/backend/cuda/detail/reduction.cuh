#pragma once

#include <cg/runtime/cuda/detail/launch.hpp>

#include <cub/block/block_reduce.cuh>

namespace cg
{
namespace cuda
{
namespace detail
{

inline constexpr auto block_reduction_algorithm = cub::BLOCK_REDUCE_WARP_REDUCTIONS;
using BlockReduction = cub::BlockReduce<double, block_size, block_reduction_algorithm>;

// Collective over the entire block. Only thread 0 receives the aggregate;
// callers already consume it only on thread 0, so no broadcast is needed.
__device__ inline double block_sum(double value)
{
    __shared__ BlockReduction::TempStorage temporary;
    const double aggregate = BlockReduction(temporary).Sum(value);
    // Dense grid-stride rows reuse this shared storage in the same block.
    __syncthreads();
    return threadIdx.x == 0 ? aggregate : 0.0;
}

} // namespace detail
} // namespace cuda
} // namespace cg
