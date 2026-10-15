#pragma once

#include <cg/backend/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

#include <memory>

namespace cg
{
namespace cuda
{

class Backend;
// One active solve per workspace. The workspace owns its reduction buffers,
// pinned scalar and cuBLAS handle, all allocated before a prepared solve.
class Workspace
{
    public:
    Workspace(std::size_t size, const Context &context);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    std::size_t size() const noexcept;
    int device() const noexcept;
    std::uintptr_t stream() const noexcept;
    VectorView residual() noexcept;
    VectorView direction() noexcept;
    VectorView product() noexcept;
    VectorView scratch() noexcept;
    VectorView preconditioned() noexcept;
    bool try_acquire() noexcept;
    void release() noexcept;

    private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
    friend class Backend;
};

} // namespace cuda
} // namespace cg
