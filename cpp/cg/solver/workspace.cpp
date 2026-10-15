#include <cg/solver/workspace.hpp>

#include <algorithm>
#include <utility>

namespace cg
{

Workspace::Workspace(std::size_t size) : m_r(size), m_p(size), m_ap(size), m_scratch(size), m_z(size) {}

Workspace::Workspace(Workspace &&other) noexcept
{
    *this = std::move(other);
}

Workspace &Workspace::operator=(Workspace &&other) noexcept
{
    if (this != &other)
    {
        m_r = std::move(other.m_r);
        m_p = std::move(other.m_p);
        m_ap = std::move(other.m_ap);
        m_scratch = std::move(other.m_scratch);
        m_z = std::move(other.m_z);
        other.m_r.clear();
        other.m_p.clear();
        other.m_ap.clear();
        other.m_scratch.clear();
        other.m_z.clear();
    }
    return *this;
}

std::size_t Workspace::capacity() const noexcept
{
    return std::min({m_r.capacity(), m_p.capacity(), m_ap.capacity(), m_scratch.capacity(), m_z.capacity()});
}

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
