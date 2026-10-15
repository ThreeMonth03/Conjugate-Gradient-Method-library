#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/blas.hpp>
#include <cg/matrix/validation.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cg
{
namespace
{

void same_size(ConstVectorView x, ConstVectorView y)
{
    detail::validate(x);
    detail::validate(y);
    if (x.size() != y.size())
    {
        throw std::invalid_argument("vector dimensions do not match");
    }
}

void no_overlap(ConstVectorView x, ConstVectorView y)
{
    if (detail::overlaps(x, y))
    {
        throw std::invalid_argument("operation requires disjoint storage");
    }
}

void exact_or_disjoint(ConstVectorView input, ConstVectorView output)
{
    if (input.data() != output.data())
    {
        no_overlap(input, output);
    }
}

void validate_apply(DenseMatrixView A, ConstVectorView x, VectorView output)
{
    detail::validate(A);
    detail::validate(x);
    detail::validate(output);
    if (A.cols != x.size() || A.rows != output.size())
    {
        throw std::invalid_argument("matrix-vector dimensions do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("matrix and output storage overlap");
    }
}

double row_dot(const double *row, const double *x, std::size_t size, bool optimized)
{
    if (!optimized)
    {
        double sum = 0.0;
#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
        for (std::size_t j = 0; j < size; ++j)
        {
            sum += row[j] * x[j];
        }
        return sum;
    }
    // Independent accumulators let ordinary optimization vectorize the row
    // without fast-math, architecture flags, or alignment stronger than double.
    double sum0 = 0.0;
    double sum1 = 0.0;
    double sum2 = 0.0;
    double sum3 = 0.0;
    std::size_t j = 0;
    for (; j + 3 < size; j += 4)
    {
        sum0 += row[j] * x[j];
        sum1 += row[j + 1] * x[j + 1];
        sum2 += row[j + 2] * x[j + 2];
        sum3 += row[j + 3] * x[j + 3];
    }
    double sum = (sum0 + sum1) + (sum2 + sum3);
    for (; j < size; ++j)
    {
        sum += row[j] * x[j];
    }
    return sum;
}

} // namespace

CpuBackend::CpuBackend(int threads, bool fused, DenseKernel dense_kernel)
    : m_threads(threads), m_fused(fused), m_dense_kernel(dense_kernel)
{
    if (threads < 1)
    {
        throw std::invalid_argument("threads must be a positive integer");
    }
    if (dense_kernel != DenseKernel::native && dense_kernel != DenseKernel::blas)
    {
        throw std::invalid_argument("unsupported dense kernel");
    }
    if (dense_kernel == DenseKernel::blas)
    {
        if (!blas_enabled())
        {
            throw std::invalid_argument("BLAS dense kernel requires a BLAS-enabled build");
        }
        if (threads != 1)
        {
            throw std::invalid_argument("BLAS dense kernel requires threads=1");
        }
    }
    else if (threads > 1 && !openmp_enabled())
    {
        throw std::invalid_argument("multiple threads require an OpenMP build");
    }
}

bool CpuBackend::openmp_enabled() noexcept
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

void CpuBackend::copy(ConstVectorView source, VectorView destination) const
{
    same_size(source, destination);
    exact_or_disjoint(source, destination);
    if (source.data() != destination.data() && source.size() != 0)
    {
        std::copy_n(source.data(), source.size(), destination.data());
    }
}

double CpuBackend::dot(ConstVectorView x, ConstVectorView y) const
{
    same_size(x, y);
    double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : sum) schedule(static)                                                      \
    num_threads(m_threads) if (m_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        sum += x.data()[i] * y.data()[i];
    }
    return sum;
}

double CpuBackend::norm2(ConstVectorView x) const
{
    return dot(x, x);
}

double CpuBackend::norm(ConstVectorView x) const
{
    return detail::norm_from_squared(x, norm2(x));
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
        const double magnitude = std::abs(x.data()[i]);
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

void CpuBackend::multiply(ConstVectorView x, ConstVectorView y, VectorView output) const
{
    same_size(x, y);
    same_size(x, output);
    exact_or_disjoint(x, output);
    exact_or_disjoint(y, output);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_threads) if (m_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        output.data()[i] = x.data()[i] * y.data()[i];
    }
}

double CpuBackend::multiply_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const
{
    same_size(factors, r);
    same_size(r, z);
    no_overlap(factors, z);
    no_overlap(r, z);
    if (!m_fused)
    {
        multiply(factors, r, z);
        return dot(r, z);
    }
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : product) schedule(static)                                                  \
    num_threads(m_threads) if (m_threads > 1 && r.size() >= 16384)
#endif
    for (std::size_t i = 0; i < r.size(); ++i)
    {
        const double value = factors.data()[i] * r.data()[i];
        z.data()[i] = value;
        product += r.data()[i] * value;
    }
    return product;
}

void CpuBackend::axpy(double alpha, ConstVectorView x, VectorView y) const
{
    same_size(x, y);
    exact_or_disjoint(x, y);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_threads) if (m_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        y.data()[i] += alpha * x.data()[i];
    }
}

void CpuBackend::axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y, VectorView output) const
{
    same_size(x, y);
    same_size(x, output);
    exact_or_disjoint(x, output);
    exact_or_disjoint(y, output);
#ifdef _OPENMP
#pragma omp parallel for simd schedule(static) num_threads(m_threads) if (m_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        output.data()[i] = alpha * x.data()[i] + beta * y.data()[i];
    }
}

void CpuBackend::apply(DenseMatrixView A, ConstVectorView x, VectorView output) const
{
    validate_apply(A, x, output);
    if (m_dense_kernel == DenseKernel::blas)
    {
        detail::blas_apply(A, x, output);
        return;
    }
    if (A.cols == 0)
    {
        if (output.size() != 0)
        {
            std::fill_n(output.data(), output.size(), 0.0);
        }
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(m_threads) if (m_threads > 1 && A.rows >= 64 && A.cols >= 128)
#endif
    for (std::size_t i = 0; i < A.rows; ++i)
    {
        const auto *row = A.data + i * A.row_stride;
        output.data()[i] = row_dot(row, x.data(), A.cols, m_fused);
    }
}

double CpuBackend::apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const
{
    same_size(p, ap);
    validate_apply(A, p, ap);
    if (!m_fused || m_dense_kernel == DenseKernel::blas)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                                       \
    num_threads(m_threads) if (m_threads > 1 && A.rows >= 64 && A.cols >= 128)
#endif
    for (std::size_t i = 0; i < A.rows; ++i)
    {
        const auto *row = A.data + i * A.row_stride;
        const double sum = row_dot(row, p.data(), A.cols, true);
        ap.data()[i] = sum;
        product += p.data()[i] * sum;
    }
    return product;
}

double CpuBackend::residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                            bool b_minus_ax) const
{
    same_size(b, r);
    no_overlap(b, r);
    validate_apply(A, x, r);
    if (m_dense_kernel == DenseKernel::blas)
    {
        detail::blas_apply(A, x, r);
        if (!m_fused)
        {
            axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
            return norm2(r);
        }
        double squared_norm = 0.0;
        for (std::size_t i = 0; i < r.size(); ++i)
        {
            const double value = b_minus_ax ? b.data()[i] - r.data()[i] : r.data()[i] - b.data()[i];
            r.data()[i] = value;
            squared_norm += value * value;
        }
        return squared_norm;
    }
    if (A.cols == 0)
    {
        axpby(b_minus_ax ? 1.0 : -1.0, b, 0.0, b, r);
        return norm2(r);
    }
    if (!m_fused)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return norm2(r);
    }
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                                                  \
    num_threads(m_threads) if (m_threads > 1 && A.rows >= 64 && A.cols >= 128)
#endif
    for (std::size_t i = 0; i < A.rows; ++i)
    {
        const auto *row = A.data + i * A.row_stride;
        const double sum = row_dot(row, x.data(), A.cols, true);
        const double value = b_minus_ax ? b.data()[i] - sum : sum - b.data()[i];
        r.data()[i] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

double CpuBackend::update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                                    double alpha) const
{
    same_size(x, r);
    same_size(x, p);
    same_size(x, ap);
    no_overlap(x, r);
    no_overlap(x, p);
    no_overlap(x, ap);
    no_overlap(r, p);
    no_overlap(r, ap);
    if (!m_fused)
    {
        axpy(alpha, p, x);
        axpy(-alpha, ap, r);
        return norm2(r);
    }
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+ : squared_norm) schedule(static)                                             \
    num_threads(m_threads) if (m_threads > 1 && x.size() >= 16384)
#endif
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x.data()[i] += alpha * p.data()[i];
        r.data()[i] -= alpha * ap.data()[i];
        squared_norm += r.data()[i] * r.data()[i];
    }
    return squared_norm;
}

void CpuBackend::update_direction(ConstVectorView r, VectorView p, double beta, double residual_scale) const
{
    same_size(r, p);
    no_overlap(r, p);
    axpby(residual_scale, r, beta, p, p);
}

} // namespace cg
