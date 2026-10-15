#include <cg/solver/cuda/detail/workspace.hpp>

namespace cg
{
namespace cuda
{

Workspace::Workspace(std::size_t size, const Context &context) : m_impl(std::make_unique<Impl>(size, context)) {}
Workspace::~Workspace() = default;
std::size_t Workspace::size() const noexcept
{
    return m_impl->r.size();
}
int Workspace::device() const noexcept
{
    return m_impl->context.device();
}
std::uintptr_t Workspace::stream() const noexcept
{
    return m_impl->context.stream();
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
bool Workspace::try_acquire() noexcept
{
    return !m_impl->active.exchange(true);
}
void Workspace::release() noexcept
{
    m_impl->active.store(false);
}

} // namespace cuda
} // namespace cg
