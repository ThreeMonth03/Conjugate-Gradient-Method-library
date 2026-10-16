#pragma once

#include <cg/solver/cuda/workspace.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

struct ExecutionResources;
struct DeviceExecutionResources;

class WorkspaceUseGuard
{
    public:
    explicit WorkspaceUseGuard(Workspace &workspace);
    ~WorkspaceUseGuard();
    WorkspaceUseGuard(const WorkspaceUseGuard &) = delete;
    WorkspaceUseGuard &operator=(const WorkspaceUseGuard &) = delete;
    ExecutionResources &resources() noexcept;
    DeviceExecutionResources &device_resources();
    void complete() noexcept { m_completed = true; }

    private:
    Workspace &m_workspace;
    bool m_completed = false;
};

} // namespace detail
} // namespace cuda
} // namespace cg
