#pragma once

#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/runtime/detail/checked_size.hpp>
#include <cg/solver/cuda/detail/device_execution.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/storage/cuda/vector.hpp>

#include <atomic>

namespace cg
{
namespace cuda
{

struct Workspace::Impl
{
    Impl(std::size_t size, const Context &context, DenseKernel kernel, ExecutionMode mode)
        : storage(cg::detail::checked_product(size, 5), context), r(slice(0)), p(slice(1)), ap(slice(2)),
          scratch(slice(3)), z(slice(4)), execution(context, kernel), mode(mode)
    {
        if (mode == ExecutionMode::device)
        {
            control.emplace(context);
        }
    }
    // All solver vectors share one owner but expose disjoint borrowed spans.
    Vector storage;
    VectorView r, p, ap, scratch, z;
    detail::ExecutionResources execution;
    ExecutionMode mode;
    std::optional<detail::DeviceExecutionResources> control;
    std::atomic<bool> active{false};

    private:
    VectorView slice(std::size_t index) noexcept
    {
        const std::size_t size = storage.size() / 5;
        return {size == 0 ? nullptr : storage.data() + index * size, size, storage.device()};
    }
};

} // namespace cuda
} // namespace cg
