#pragma once

#include <cg/storage/vector_view.hpp>

#include <vector>

namespace cg
{

class Vector
{
    public:
    explicit Vector(std::size_t size = 0);
    explicit Vector(std::vector<double> values);
    std::size_t size() const noexcept { return m_values.size(); }
    std::size_t capacity() const noexcept { return m_values.capacity(); }
    double *data() noexcept { return m_values.data(); }
    const double *data() const noexcept { return m_values.data(); }
    VectorView view() noexcept { return {data(), size()}; }
    ConstVectorView view() const noexcept { return {data(), size()}; }
    void resize(std::size_t size);

    private:
    std::vector<double> m_values;
};

} // namespace cg
