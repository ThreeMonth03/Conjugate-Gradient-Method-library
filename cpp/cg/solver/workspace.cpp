#include <cg/solver/workspace.hpp>

namespace cg
{

Workspace::Workspace(std::size_t size) : m_r(size), m_p(size), m_ap(size), m_scratch(size), m_z(size) {}

void Workspace::prepare(std::size_t size)
{
    if (size > capacity())
    {
        *this = Workspace(size);
        return;
    }
    m_r.resize(size);
    m_p.resize(size);
    m_ap.resize(size);
    m_scratch.resize(size);
    m_z.resize(size);
}

} // namespace cg
