#include <cg/backend/cuda/detail/runtime.hpp>
#include <cg/storage/cuda/validation.hpp>
#include <cg/storage/cuda/vector.hpp>
#include <cg/storage/validation.hpp>

#include <utility>
#include <vector>

namespace cg
{
namespace cuda
{

Vector::Vector(std::size_t size, const Context &context) : m_size(size), m_device(context.device())
{
    const auto bytes = detail::product(size, sizeof(double));
    detail::DeviceGuard guard(m_device);
    if (bytes != 0)
    {
        detail::check(cudaMalloc(reinterpret_cast<void **>(&m_data), bytes), "cudaMalloc");
    }
}
Vector::Vector(cg::ConstVectorView host, const Context &context) : Vector(host.size, context)
{
    upload(host, context);
}
Vector::~Vector()
{
    if (m_data != nullptr)
    {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(m_device);
        (void)cudaFree(m_data);
        (void)cudaSetDevice(previous);
    }
}
Vector::Vector(Vector &&other) noexcept
    : m_data(std::exchange(other.m_data, nullptr)), m_size(std::exchange(other.m_size, 0)),
      m_device(std::exchange(other.m_device, -1))
{
}
Vector &Vector::operator=(Vector &&other) noexcept
{
    if (this != &other)
    {
        Vector temporary(std::move(other));
        std::swap(m_data, temporary.m_data);
        std::swap(m_size, temporary.m_size);
        std::swap(m_device, temporary.m_device);
    }
    return *this;
}
void Vector::upload(cg::ConstVectorView host, const Context &context)
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size != m_size)
    {
        throw std::invalid_argument("CUDA upload dimensions do not match");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0)
    {
        detail::check(cudaMemcpyAsync(m_data, host.data, detail::product(m_size, sizeof(double)),
                                      cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpyAsync upload");
    }
    context.synchronize();
}
void Vector::download(cg::VectorView host, const Context &context) const
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size != m_size)
    {
        throw std::invalid_argument("CUDA download dimensions do not match");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0)
    {
        detail::check(cudaMemcpyAsync(host.data, m_data, detail::product(m_size, sizeof(double)),
                                      cudaMemcpyDeviceToHost, detail::stream(context)),
                      "cudaMemcpyAsync download");
    }
    context.synchronize();
}
void Vector::copy_from_device(ConstVectorView source, const Context &context)
{
    detail::validate(source, context);
    detail::validate(view(), context);
    if (source.size != m_size || (source.data != m_data && detail::overlaps(source, view())))
    {
        throw std::invalid_argument("CUDA device copy dimensions or storage overlap");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0 && source.data != m_data)
    {
        detail::check(cudaMemcpyAsync(m_data, source.data, detail::product(m_size, sizeof(double)),
                                      cudaMemcpyDeviceToDevice, detail::stream(context)),
                      "cudaMemcpyAsync device copy");
    }
    context.synchronize();
}
void Vector::fill(double value, const Context &context)
{
    detail::validate(view(), context);
    detail::fill(view(), value, context);
    context.synchronize();
}

} // namespace cuda
} // namespace cg
