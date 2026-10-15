#pragma once

#include <cg/storage/vector.hpp>

namespace cg
{

// One workspace belongs to one active solve. Reuse it between calls; distinct
// workspaces are required for simultaneous solves, including on one backend.
class Workspace
{
    public:
    explicit Workspace(std::size_t size = 0);
    void prepare(std::size_t size);
    std::size_t size() const noexcept { return m_r.size(); }
    std::size_t capacity() const noexcept { return m_r.capacity(); }
    VectorView residual() noexcept { return m_r.view(); }
    VectorView direction() noexcept { return m_p.view(); }
    VectorView product() noexcept { return m_ap.view(); }
    VectorView scratch() noexcept { return m_scratch.view(); }
    VectorView preconditioned() noexcept { return m_z.view(); }

    private:
    Vector m_r;
    Vector m_p;
    Vector m_ap;
    Vector m_scratch;
    Vector m_z;
};

} // namespace cg
