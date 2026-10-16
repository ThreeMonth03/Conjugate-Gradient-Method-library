#pragma once

#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/device_memory.hpp>
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
    int device() const noexcept { return m_data.get_deleter().device; }
    // Pointers and views borrow the allocation; only live owners provide them.
    double *data() & noexcept { return m_data.get(); }
    const double *data() const & noexcept { return m_data.get(); }
    double *data() && = delete;
    const double *data() const && = delete;
    VectorView view() & noexcept { return {data(), size(), device()}; }
    ConstVectorView view() const & noexcept { return {data(), size(), device()}; }
    VectorView view() && = delete;
    ConstVectorView view() const && = delete;
    void upload(cg::ConstVectorView host, const Context &context);
    void download(cg::VectorView host, const Context &context) const;
    void copy_from_device(ConstVectorView source, const Context &context);
    void fill(double value, const Context &context);

    private:
    detail::DeviceAllocation<double> m_data;
    std::size_t m_size = 0;
};

} // namespace cuda
} // namespace cg
