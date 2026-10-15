#pragma once

#include <cg/backend/cuda/detail/blas_handle.hpp>
#include <cg/backend/cuda/detail/launch.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/storage/cuda/detail/allocation.hpp>
#include <cg/storage/cuda/vector.hpp>

#include <atomic>

namespace cg
{
namespace cuda
{
namespace detail
{

struct CublasResources
{
    explicit CublasResources(const Context &context)
        : workspace(device_allocation<unsigned char>(workspace_bytes, context)), handle(context)
    {
        DeviceGuard guard(context.device());
        check(cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        check(cublasSetAtomicsMode(handle.value, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        check(cublasSetWorkspace(handle.value, workspace.get(), workspace_bytes), "cublasSetWorkspace");
    }
    static constexpr std::size_t workspace_bytes = 4 * 1024 * 1024;
    DeviceAllocation<unsigned char> workspace;
    BlasHandle handle;
};

} // namespace detail

struct Workspace::Impl
{
    Impl(std::size_t size, const Context &ctx, DenseKernel selected)
        : context(ctx), r(size, ctx), p(size, ctx), ap(size, ctx), scratch(size, ctx), z(size, ctx),
          partials(detail::device_allocation<double>(detail::max_blocks, ctx)),
          scalar(detail::device_allocation<double>(1, ctx)), host(detail::pinned_scalar()), kernel(selected)
    {
        if (selected != DenseKernel::native && selected != DenseKernel::cublas)
        {
            throw std::invalid_argument("unsupported CUDA dense kernel");
        }
        if (selected == DenseKernel::cublas)
        {
            cublas = std::make_unique<detail::CublasResources>(ctx);
        }
    }
    Context context;
    Vector r, p, ap, scratch, z;
    detail::DeviceAllocation<double> partials, scalar;
    detail::PinnedScalar host;
    std::unique_ptr<detail::CublasResources> cublas;
    DenseKernel kernel;
    std::atomic<bool> active{false};
};

} // namespace cuda
} // namespace cg
