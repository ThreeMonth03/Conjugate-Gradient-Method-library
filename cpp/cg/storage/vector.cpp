#include <cg/storage/vector.hpp>

#include <utility>

namespace cg
{

Vector::Vector(std::size_t size) : m_values(size) {}
Vector::Vector(std::vector<double> values) : m_values(std::move(values)) {}
void Vector::resize(std::size_t size)
{
    m_values.resize(size);
}

} // namespace cg
