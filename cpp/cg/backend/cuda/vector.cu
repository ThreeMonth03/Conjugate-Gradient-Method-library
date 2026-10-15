#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/backend/cuda/detail/reduction.cuh>
#include <cg/backend/cuda/detail/validation.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{
namespace
{

__global__ void linear_kernel(ConstVectorView x, ConstVectorView y, VectorView output, double alpha, double beta)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < output.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        output.data()[i] = alpha * x.data()[i] + beta * y.data()[i];
    }
}

__global__ void multiply_kernel(ConstVectorView factors, ConstVectorView r, VectorView z, double *partials, bool reduce)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < r.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double value = factors.data()[i] * r.data()[i];
        z.data()[i] = value;
        if (reduce)
        {
            sum += r.data()[i] * value;
        }
    }
    const double total = detail::block_sum(sum);
    if (threadIdx.x == 0 && reduce)
    {
        partials[blockIdx.x] = total;
    }
}

__global__ void update_kernel(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap, double alpha,
                              double *partials)
{
    double sum = 0.0;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        x.data()[i] += alpha * p.data()[i];
        const double value = r.data()[i] - alpha * ap.data()[i];
        r.data()[i] = value;
        sum += value * value;
    }
    const double total = detail::block_sum(sum);
    if (threadIdx.x == 0)
    {
        partials[blockIdx.x] = total;
    }
}

} // namespace

void Backend::copy(ConstVectorView source, VectorView destination) const
{
    detail::same_size(source, destination, m_resources.context);
    detail::exact_or_disjoint(source, destination);
    detail::DeviceGuard guard(m_resources.context.device());
    if (source.size() != 0 && source.data() != destination.data())
    {
        detail::check(cudaMemcpyAsync(destination.data(), source.data(), source.size() * sizeof(double),
                                      cudaMemcpyDeviceToDevice, detail::stream(m_resources.context)),
                      "cudaMemcpyAsync copy");
    }
}

double Backend::multiply_and_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const
{
    detail::same_size(factors, r, m_resources.context);
    detail::same_size(r, z, m_resources.context);
    detail::no_overlap(factors, z);
    detail::no_overlap(r, z);
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::blocks(r.size());
    multiply_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(
        factors, r, z, m_resources.partials.get(), m_fused);
    detail::launch_check();
    return m_fused ? finish_reduction(blocks) : dot(r, z);
}

void Backend::axpy(double alpha, ConstVectorView x, VectorView y) const
{
    detail::same_size(x, y, m_resources.context);
    detail::exact_or_disjoint(x, y);
    detail::DeviceGuard guard(m_resources.context.device());
    linear_kernel<<<detail::blocks(x.size()), detail::block_size, 0, detail::stream(m_resources.context)>>>(x, y, y,
                                                                                                            alpha, 1.0);
    detail::launch_check();
}

void Backend::axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const
{
    detail::same_size(x, y, m_resources.context);
    detail::same_size(x, output, m_resources.context);
    detail::exact_or_disjoint(x, output);
    detail::exact_or_disjoint(y, output);
    detail::DeviceGuard guard(m_resources.context.device());
    linear_kernel<<<detail::blocks(x.size()), detail::block_size, 0, detail::stream(m_resources.context)>>>(
        x, y, output, alpha, beta);
    detail::launch_check();
}

double Backend::update_x_r_squared_norm(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                                        double alpha) const
{
    detail::same_size(x, r, m_resources.context);
    detail::same_size(x, p, m_resources.context);
    detail::same_size(x, ap, m_resources.context);
    detail::no_overlap(x, r);
    detail::no_overlap(x, p);
    detail::no_overlap(x, ap);
    detail::no_overlap(r, p);
    detail::no_overlap(r, ap);
    if (!m_fused)
    {
        axpy(alpha, p, x);
        axpy(-alpha, ap, r);
        return dot(r, r);
    }
    detail::DeviceGuard guard(m_resources.context.device());
    const auto blocks = detail::blocks(x.size());
    update_kernel<<<blocks, detail::block_size, 0, detail::stream(m_resources.context)>>>(x, r, p, ap, alpha,
                                                                                          m_resources.partials.get());
    detail::launch_check();
    return finish_reduction(blocks);
}

void Backend::update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale) const
{
    detail::no_overlap(r, p);
    axpby(residual_scale, r, beta, p, p);
}

} // namespace detail
} // namespace cuda
} // namespace cg
