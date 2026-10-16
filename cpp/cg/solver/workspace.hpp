#pragma once

#include <cg/storage/vector_view.hpp>

#include <vector>

namespace cg
{

// One workspace belongs to one active solve. Reuse it between calls; distinct
// workspaces are required for simultaneous solves, including on one backend.
// Moving transfers all buffers and leaves an empty, reusable workspace.
class Workspace
{
    public:
    explicit Workspace(std::size_t size = 0);
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    Workspace(Workspace &&other) noexcept;
    Workspace &operator=(Workspace &&other) noexcept;
    // Growth commits all buffers together. Requests within capacity allocate
    // nothing; an allocation failure leaves every buffer and value unchanged.
    void prepare(std::size_t size);
    std::size_t size() const noexcept { return m_r.size(); }
    std::size_t capacity() const noexcept;
    // Buffer views borrow this workspace and cannot outlive its current storage.
    VectorView residual() & noexcept { return m_r; }
    VectorView residual() && = delete;
    VectorView direction() & noexcept { return m_p; }
    VectorView direction() && = delete;
    VectorView product() & noexcept { return m_ap; }
    VectorView product() && = delete;
    VectorView scratch() & noexcept { return m_scratch; }
    VectorView scratch() && = delete;
    VectorView preconditioned() & noexcept { return m_z; }
    VectorView preconditioned() && = delete;

    private:
    std::vector<double> m_r;
    std::vector<double> m_p;
    std::vector<double> m_ap;
    std::vector<double> m_scratch;
    std::vector<double> m_z;
};

} // namespace cg
