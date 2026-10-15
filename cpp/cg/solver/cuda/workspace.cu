#include <cg/solver/cuda/detail/workspace.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>

namespace cg
{
namespace cuda
{

Workspace::Workspace(std::size_t size, const Context &context, DenseKernel kernel)
    : m_impl(std::make_unique<Impl>(size, context, kernel))
{
}
Workspace::~Workspace() = default;
std::size_t Workspace::size() const noexcept
{
    return m_impl->r.size();
}
int Workspace::device() const noexcept
{
    return m_impl->execution.context.device();
}
std::uintptr_t Workspace::stream() const noexcept
{
    return m_impl->execution.context.stream();
}
DenseKernel Workspace::dense_kernel() const noexcept
{
    return m_impl->execution.kernel;
}
VectorView Workspace::residual() noexcept
{
    return m_impl->r.view();
}
VectorView Workspace::direction() noexcept
{
    return m_impl->p.view();
}
VectorView Workspace::product() noexcept
{
    return m_impl->ap.view();
}
VectorView Workspace::scratch() noexcept
{
    return m_impl->scratch.view();
}
VectorView Workspace::preconditioned() noexcept
{
    return m_impl->z.view();
}
const Context &Workspace::execution_context() const noexcept
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

WorkspaceUseGuard::WorkspaceUseGuard(Workspace &workspace, const Context &context)
    : m_workspace(workspace), m_context(context)
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
            m_context.synchronize();
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

} // namespace detail

} // namespace cuda
} // namespace cg
