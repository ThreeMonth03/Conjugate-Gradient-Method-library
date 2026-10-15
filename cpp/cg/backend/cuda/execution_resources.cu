#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

CublasResources::CublasResources(const Context &context)
    : workspace(device_allocation<unsigned char>(workspace_bytes, context)), handle(context)
{
    DeviceGuard guard(context.device());
    check(cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
    check(cublasSetAtomicsMode(handle.value, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
    check(cublasSetWorkspace(handle.value, workspace.get(), workspace_bytes), "cublasSetWorkspace");
}
ExecutionResources::ExecutionResources(const Context &ctx, DenseKernel selected)
    : context(ctx), partials(device_allocation<double>(max_blocks, ctx)), scalar(device_allocation<double>(1, ctx)),
      host(pinned_scalar()), kernel(selected)
{
    if (selected != DenseKernel::native && selected != DenseKernel::cublas)
    {
        throw std::invalid_argument("unsupported CUDA dense kernel");
    }
    if (selected == DenseKernel::cublas)
    {
        cublas.emplace(ctx);
    }
}

} // namespace detail
} // namespace cuda
} // namespace cg
