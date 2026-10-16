#pragma once

#include <cg/backend/cuda/dense_kernel.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

#include <memory>

namespace cg
{
namespace cuda
{

enum class ExecutionMode
{
    host,
    device
};

namespace detail
{
class WorkspaceUseGuard;
}
// One active solve per workspace. Its execution context, dense kernel, and
// execution mode are fixed at construction. Reduction buffers, inline host
// scalar, and optional cuBLAS resources are allocated before any solve.
class Workspace
{
    public:
    Workspace(std::size_t size, const Context &context, DenseKernel kernel = DenseKernel::native,
              ExecutionMode mode = ExecutionMode::host);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    std::size_t size() const noexcept;
    int device() const noexcept;
    std::uintptr_t stream() const noexcept;
    DenseKernel dense_kernel() const noexcept;
    ExecutionMode execution_mode() const noexcept;
    // Borrowed execution context; valid for this workspace's lifetime.
    const Context &context() const & noexcept;
    const Context &context() const && = delete;
    VectorView residual() & noexcept;
    VectorView residual() && = delete;
    VectorView direction() & noexcept;
    VectorView direction() && = delete;
    VectorView product() & noexcept;
    VectorView product() && = delete;
    VectorView scratch() & noexcept;
    VectorView scratch() && = delete;
    VectorView preconditioned() & noexcept;
    VectorView preconditioned() && = delete;

    private:
    bool try_acquire() noexcept;
    void release() noexcept;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    friend class detail::WorkspaceUseGuard;
};

} // namespace cuda
} // namespace cg
