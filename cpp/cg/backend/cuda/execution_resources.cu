#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{
namespace
{

std::size_t reduction_offset(DenseKernel selected)
{
    switch (selected)
    {
    case DenseKernel::native:
        return 0;
    case DenseKernel::cublas:
        return CublasResources::workspace_bytes;
    }
    throw std::invalid_argument("unsupported CUDA dense kernel");
}

static_assert(CublasResources::workspace_bytes % 256 == 0);
static_assert(CublasResources::workspace_bytes % alignof(double) == 0);

} // namespace

CublasResources::CublasResources(const Context &context, void *workspace) : handle(blas_handle(context))
{
    DeviceGuard guard(context.device());
    check(cublasSetPointerMode(handle.get(), CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    check(cublasSetAtomicsMode(handle.get(), CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
    check(cublasSetWorkspace(handle.get(), workspace, workspace_bytes), "cublasSetWorkspace");
}
ExecutionResources::ExecutionResources(const Context &ctx, DenseKernel selected)
    : context(ctx),
      storage(device_allocation<std::byte>(reduction_offset(selected) + (max_blocks + 1) * sizeof(double), ctx)),
      partials(reinterpret_cast<double *>(storage.get() + reduction_offset(selected))), scalar(partials + max_blocks)
{
    if (selected == DenseKernel::cublas)
    {
        cublas.emplace(ctx, storage.get());
    }
}

} // namespace detail
} // namespace cuda
} // namespace cg
