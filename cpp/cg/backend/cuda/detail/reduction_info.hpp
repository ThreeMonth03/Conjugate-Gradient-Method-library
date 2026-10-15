#pragma once

namespace cg
{
namespace cuda
{
namespace detail
{

struct BlockReductionInfo
{
    int cub_version;
    unsigned int block_threads;
    const char *algorithm;
};

// Defined in the CUDA translation unit that instantiates BlockReduction.
// This query does not require an available GPU or initialize CUDA resources.
BlockReductionInfo block_reduction_info() noexcept;

} // namespace detail
} // namespace cuda
} // namespace cg
