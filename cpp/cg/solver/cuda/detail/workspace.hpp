#pragma once

#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/storage/cuda/vector.hpp>

#include <atomic>

namespace cg
{
namespace cuda
{

struct Workspace::Impl
{
    Impl(std::size_t size, const Context &context, DenseKernel kernel)
        : r(size, context), p(size, context), ap(size, context), scratch(size, context), z(size, context),
          execution(context, kernel)
    {
    }
    Vector r, p, ap, scratch, z;
    detail::ExecutionResources execution;
    std::atomic<bool> active{false};
};

} // namespace cuda
} // namespace cg
