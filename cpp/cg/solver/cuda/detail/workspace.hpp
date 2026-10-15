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

struct Workspace::Impl
{
    Impl(std::size_t size, const Context &ctx)
        : context(ctx), r(size, ctx), p(size, ctx), ap(size, ctx), scratch(size, ctx), z(size, ctx),
          partials(detail::max_blocks, ctx), scalar(1, ctx), blas_workspace(4 * 1024 * 1024, ctx), handle(ctx)
    {
        detail::DeviceGuard guard(ctx.device());
        detail::check(cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
        detail::check(cublasSetAtomicsMode(handle.value, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
        detail::check(cublasSetWorkspace(handle.value, blas_workspace.data(), 4 * 1024 * 1024), "cublasSetWorkspace");
    }
    Context context;
    Vector r, p, ap, scratch, z;
    detail::DeviceAllocation<double> partials, scalar;
    detail::DeviceAllocation<unsigned char> blas_workspace;
    detail::PinnedScalar host;
    detail::BlasHandle handle;
    std::atomic<bool> active{false};
};

} // namespace cuda
} // namespace cg
