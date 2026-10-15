#pragma once

#include <cg/backend/cuda/dense_kernel.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

#include <memory>

namespace cg
{
namespace cuda
{

class PreparedSolver;
namespace detail
{
class WorkspaceUseGuard;
}
// One active solve per workspace. The workspace owns its reduction buffers,
// inline host scalar, and optional cuBLAS resources, allocated before any solve.
class Workspace
{
    public:
    Workspace(std::size_t size, const Context &context, DenseKernel kernel = DenseKernel::native);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    std::size_t size() const noexcept;
    int device() const noexcept;
    std::uintptr_t stream() const noexcept;
    DenseKernel dense_kernel() const noexcept;
    VectorView residual() noexcept;
    VectorView direction() noexcept;
    VectorView product() noexcept;
    VectorView scratch() noexcept;
    VectorView preconditioned() noexcept;

    private:
    const Context &execution_context() const noexcept;
    bool try_acquire() noexcept;
    void release() noexcept;
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    friend class detail::WorkspaceUseGuard;
    friend class PreparedSolver;
};

} // namespace cuda
} // namespace cg
