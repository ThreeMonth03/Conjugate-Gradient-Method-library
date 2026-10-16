#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/detail/validation.hpp>

#include <algorithm>

namespace cg
{

void CpuBackend::copy(ConstVectorView source, VectorView destination) const
{
    detail::same_size(source, destination);
    detail::exact_or_disjoint(source, destination);
    if (source.data() != destination.data() && source.size() != 0)
    {
        std::ranges::copy(source, destination.begin());
    }
}

void CpuBackend::multiply(ConstVectorView x, ConstVectorView y, VectorView output) const
{
    detail::same_size(x, y);
    detail::same_size(x, output);
    detail::exact_or_disjoint(x, output);
    detail::exact_or_disjoint(y, output);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_num_threads) if (m_num_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        output[i] = x[i] * y[i];
    }
}

double CpuBackend::multiply_and_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const
{
    detail::same_size(factors, r);
    detail::same_size(r, z);
    detail::no_overlap(factors, z);
    detail::no_overlap(r, z);
    if (!m_fused)
    {
        multiply(factors, r, z);
        return dot(r, z);
    }
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : product) schedule(static)                                                  \
    num_threads(m_num_threads) if (m_num_threads > 1 && r.size() >= 16384)
#endif
    for (std::size_t i = 0; i < r.size(); ++i)
    {
        const double value = factors[i] * r[i];
        z[i] = value;
        product += r[i] * value;
    }
    return product;
}

void CpuBackend::axpy(double alpha, ConstVectorView x, VectorView y) const
{
    detail::same_size(x, y);
    detail::exact_or_disjoint(x, y);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_num_threads) if (m_num_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        y[i] += alpha * x[i];
    }
}

void CpuBackend::axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const
{
    detail::same_size(x, y);
    detail::same_size(x, output);
    detail::exact_or_disjoint(x, output);
    detail::exact_or_disjoint(y, output);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_num_threads) if (m_num_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        output[i] = alpha * x[i] + beta * y[i];
    }
}

double CpuBackend::update_x_r_squared_norm(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                                           double alpha) const
{
    detail::same_size(x, r);
    detail::same_size(x, p);
    detail::same_size(x, ap);
    detail::no_overlap(x, r);
    detail::no_overlap(x, p);
    detail::no_overlap(x, ap);
    detail::no_overlap(r, p);
    detail::no_overlap(r, ap);
    if (!m_fused)
    {
        axpy(alpha, p, x);
        axpy(-alpha, ap, r);
        return squared_norm(r);
    }
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : squared_norm) schedule(static)                                             \
    num_threads(m_num_threads) if (m_num_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] += alpha * p[i];
        r[i] -= alpha * ap[i];
        squared_norm += r[i] * r[i];
    }
    return squared_norm;
}

void CpuBackend::update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale) const
{
    detail::same_size(r, p);
    detail::no_overlap(r, p);
    axpby(residual_scale, r, beta, p, p);
}

} // namespace cg
