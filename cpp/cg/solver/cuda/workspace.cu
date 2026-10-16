#include <cg/solver/cuda/detail/workspace.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>

namespace cg
{
namespace cuda
{

Workspace::Workspace(std::size_t size, const Context &context, DenseKernel kernel, ExecutionMode mode)
{
    if (mode != ExecutionMode::host && mode != ExecutionMode::device)
    {
        throw std::invalid_argument("unsupported CUDA execution mode");
    }
    m_impl = std::make_unique<Impl>(size, context, kernel, mode);
}
Workspace::~Workspace() = default;
std::size_t Workspace::size() const noexcept
{
    return m_impl->r.size();
}
int Workspace::device() const noexcept
{
    return context().device();
}
std::uintptr_t Workspace::stream() const noexcept
{
    return context().stream();
}
DenseKernel Workspace::dense_kernel() const noexcept
{
    return m_impl->execution.dense_kernel();
}
ExecutionMode Workspace::execution_mode() const noexcept
{
    return m_impl->control ? ExecutionMode::device : ExecutionMode::host;
}
VectorView Workspace::residual() & noexcept
{
    return m_impl->r;
}
VectorView Workspace::direction() & noexcept
{
    return m_impl->p;
}
VectorView Workspace::product() & noexcept
{
    return m_impl->ap;
}
VectorView Workspace::scratch() & noexcept
{
    return m_impl->scratch;
}
VectorView Workspace::preconditioned() & noexcept
{
    return m_impl->z;
}
const Context &Workspace::context() const & noexcept
{
    return m_impl->execution.context;
}
bool Workspace::try_acquire() noexcept
{
    return !m_impl->active.exchange(true);
}
void Workspace::release() noexcept
{
    m_impl->active.store(false);
}

namespace detail
{

WorkspaceUseGuard::WorkspaceUseGuard(Workspace &workspace) : m_workspace(workspace)
{
    if (!workspace.try_acquire())
    {
        throw std::invalid_argument("CUDA workspace is already in use by another solve");
    }
}
WorkspaceUseGuard::~WorkspaceUseGuard()
{
    if (!m_completed)
    {
        try
        {
            m_workspace.context().synchronize();
        }
        catch (...)
        {
        }
    }
    m_workspace.release();
}
ExecutionResources &WorkspaceUseGuard::resources() noexcept
{
    return m_workspace.m_impl->execution;
}
DeviceExecutionResources &WorkspaceUseGuard::device_resources()
{
    return m_workspace.m_impl->control.value();
}

} // namespace detail

} // namespace cuda
} // namespace cg
