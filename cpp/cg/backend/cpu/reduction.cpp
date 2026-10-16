#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/detail/reduction.hpp>
#include <cg/backend/cpu/detail/validation.hpp>

#include <cmath>
#include <limits>

namespace cg
{

double CpuBackend::dot(ConstVectorView x, ConstVectorView y) const
{
    detail::same_size(x, y);
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : sum) schedule(static)                                                      \
    num_threads(m_num_threads) if (m_num_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sum += x[i] * y[i];
    }
    return sum;
}

double CpuBackend::squared_norm(ConstVectorView x) const
{
    return dot(x, x);
}

double CpuBackend::norm(ConstVectorView x) const
{
    return detail::norm_from_squared(x, squared_norm(x));
}

namespace detail
{

double norm_from_squared(ConstVectorView x, double squared_norm)
{
    if (std::isfinite(squared_norm) && squared_norm >= std::numeric_limits<double>::min())
    {
        return std::sqrt(squared_norm);
    }
    // The fallback is needed only outside the ordinary squared-sum range.
    // Scaling prevents both overflow and underflow in the norm evaluation.
    double scale = 0.0;
    double sum = 1.0;
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        const double magnitude = std::abs(x[i]);
        if (!std::isfinite(magnitude))
        {
            return magnitude;
        }
        if (magnitude != 0.0)
        {
            if (scale < magnitude)
            {
                const double ratio = scale / magnitude;
                sum = 1.0 + sum * ratio * ratio;
                scale = magnitude;
            }
            else
            {
                const double ratio = magnitude / scale;
                sum += ratio * ratio;
            }
        }
    }
    return scale * std::sqrt(sum);
}

} // namespace detail

} // namespace cg
