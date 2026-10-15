#include <cg/backend/cuda/backend.hpp>

#include <stdexcept>

namespace cg
{
namespace cuda
{
Backend::Backend(const Context &context, Workspace &workspace, bool fused, DenseKernel kernel)
    : m_context(context), m_workspace(workspace), m_fused(fused), m_kernel(kernel)
{
    if (workspace.device() != context.device() || workspace.stream() != context.stream())
    {
        throw std::invalid_argument("CUDA workspace must match the device and stream");
    }
    if (kernel != DenseKernel::native && kernel != DenseKernel::cublas)
    {
        throw std::invalid_argument("unsupported CUDA dense kernel");
    }
    if (workspace.dense_kernel() != kernel)
    {
        throw std::invalid_argument("CUDA workspace dense kernel does not match selected kernel");
    }
}

} // namespace cuda
} // namespace cg
