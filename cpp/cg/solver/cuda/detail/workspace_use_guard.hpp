#pragma once

#include <cg/solver/cuda/workspace.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

struct ExecutionResources;

class WorkspaceUseGuard
{
    public:
    WorkspaceUseGuard(Workspace &workspace, const Context &context);
    ~WorkspaceUseGuard();
    WorkspaceUseGuard(const WorkspaceUseGuard &) = delete;
    WorkspaceUseGuard &operator=(const WorkspaceUseGuard &) = delete;
    ExecutionResources &resources() noexcept;
    void complete() noexcept { m_completed = true; }

    private:
    Workspace &m_workspace;
    const Context &m_context;
    bool m_completed = false;
};

} // namespace detail
} // namespace cuda
} // namespace cg
