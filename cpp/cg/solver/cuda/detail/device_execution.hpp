#pragma once

#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/solver/options.hpp>
#include <cg/solver/solve_info.hpp>
#include <cg/storage/cuda/device_memory.hpp>

#include <cstdint>

namespace cg
{
namespace cuda
{
namespace detail
{

enum class IterationDecision
{
    proceed,
    verify_true,
    stopped
};
struct IterationPacket
{
    Status status = Status::max_iterations;
    IterationDecision decision = IterationDecision::proceed;
    std::uint64_t advanced = 0;
    double residual_norm = 0.0;
};
struct DeviceState;

// Only device-mode workspaces allocate this state. Each solve borrows it under
// WorkspaceUseGuard; no handles, packets or device buffers are cached globally.
struct DeviceExecutionResources
{
    explicit DeviceExecutionResources(const Context &context);
    ~DeviceExecutionResources();
    DeviceAllocation<DeviceState> state;
    // Pageable storage is sufficient: every copy is synchronized before read
    // or reuse, and the exclusive workspace owner outlives pending operations.
    IterationPacket host;
};

SolveInfo solve_device(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Backend &backend,
                       ExecutionResources &execution, DeviceExecutionResources &control, const CgOptions &options,
                       bool fused, const JacobiPreconditioner *M);
SolveInfo solve_device(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                       const Backend &backend, ExecutionResources &execution, DeviceExecutionResources &control,
                       const CgOptions &options, bool fused, const JacobiPreconditioner *M);

} // namespace detail
} // namespace cuda
} // namespace cg
