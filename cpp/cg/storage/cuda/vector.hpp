#pragma once

#include <cg/backend/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>
#include <cg/storage/vector_view.hpp>

namespace cg
{
namespace cuda
{

class Vector
{
    public:
    Vector() = default;
    Vector(std::size_t size, const Context &context);
    Vector(cg::ConstVectorView host, const Context &context);
    ~Vector();
    Vector(const Vector &) = delete;
    Vector &operator=(const Vector &) = delete;
    Vector(Vector &&other) noexcept;
    Vector &operator=(Vector &&other) noexcept;
    std::size_t size() const noexcept { return m_size; }
    int device() const noexcept { return m_device; }
    double *data() noexcept { return m_data; }
    const double *data() const noexcept { return m_data; }
    VectorView view() noexcept { return {m_data, m_size, m_device}; }
    ConstVectorView view() const noexcept { return {m_data, m_size, m_device}; }
    void upload(cg::ConstVectorView host, const Context &context);
    void download(cg::VectorView host, const Context &context) const;
    void copy_from_device(ConstVectorView source, const Context &context);
    void fill(double value, const Context &context);

    private:
    double *m_data = nullptr;
    std::size_t m_size = 0;
    int m_device = -1;
};

} // namespace cuda
} // namespace cg
