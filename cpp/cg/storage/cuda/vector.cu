#include <cg/runtime/cuda/detail/launch.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/storage/cuda/detail/fill.hpp>
#include <cg/storage/cuda/detail/validation.hpp>
#include <cg/storage/cuda/vector.hpp>
#include <cg/storage/detail/validation.hpp>

#include <utility>

namespace cg
{
namespace cuda
{
namespace
{

__global__ void fill_kernel(VectorView output, double value)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        output.data()[i] = value;
    }
}

} // namespace

namespace detail
{

void fill(VectorView output, double value, const Context &context)
{
    DeviceGuard guard(context.device());
    fill_kernel<<<blocks(output.size()), block_size, 0, stream(context)>>>(output, value);
    detail::launch_check();
}

} // namespace detail

Vector::Vector(std::size_t size, const Context &context)
    : m_data(detail::device_allocation<double>(size, context)), m_size(size)
{
}
Vector::Vector(cg::ConstVectorView host, const Context &context) : Vector(host.size(), context)
{
    upload(host, context);
}
Vector::~Vector() = default;
Vector::Vector(Vector &&other) noexcept : m_data(std::move(other.m_data)), m_size(std::exchange(other.m_size, 0))
{
    other.m_data.get_deleter().device = -1;
}
Vector &Vector::operator=(Vector &&other) noexcept
{
    if (this != &other)
    {
        Vector temporary(std::move(other));
        m_data.swap(temporary.m_data);
        std::swap(m_size, temporary.m_size);
    }
    return *this;
}
void Vector::upload(cg::ConstVectorView host, const Context &context)
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size() != m_size)
    {
        throw std::invalid_argument("CUDA upload dimensions do not match");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0)
    {
        detail::check(cudaMemcpyAsync(data(), host.data(), detail::checked_product(m_size, sizeof(double)),
                                      cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpyAsync upload");
    }
    context.synchronize();
}
void Vector::download(cg::VectorView host, const Context &context) const
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size() != m_size)
    {
        throw std::invalid_argument("CUDA download dimensions do not match");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0)
    {
        detail::check(cudaMemcpyAsync(host.data(), data(), detail::checked_product(m_size, sizeof(double)),
                                      cudaMemcpyDeviceToHost, detail::stream(context)),
                      "cudaMemcpyAsync download");
    }
    context.synchronize();
}
void Vector::copy_from_device(ConstVectorView source, const Context &context)
{
    detail::validate(source, context);
    detail::validate(view(), context);
    if (source.size() != m_size || (source.data() != data() && detail::overlaps(source, view())))
    {
        throw std::invalid_argument("CUDA device copy dimensions or storage overlap");
    }
    detail::DeviceGuard guard(context.device());
    if (m_size != 0 && source.data() != data())
    {
        detail::check(cudaMemcpyAsync(data(), source.data(), detail::checked_product(m_size, sizeof(double)),
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
