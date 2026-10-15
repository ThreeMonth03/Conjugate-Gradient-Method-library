#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/detail/blas.hpp>
#include <cg/backend/cpu/detail/validation.hpp>
#include <cg/matrix/detail/validation.hpp>

#include <algorithm>

namespace cg
{
namespace
{

void validate_apply(DenseMatrixView A, ConstVectorView x, VectorView output)
{
    detail::validate(A);
    detail::validate(x);
    detail::validate(output);
    if (A.cols() != x.size() || A.rows() != output.size())
    {
        throw std::invalid_argument("matrix-vector dimensions do not match");
    }
    detail::no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("matrix and output storage overlap");
    }
}

double row_dot(ConstVectorView row, ConstVectorView x, bool optimized)
{
    const auto size = row.size();
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

void CpuBackend::apply(DenseMatrixView A, ConstVectorView x, VectorView output) const
{
    validate_apply(A, x, output);
    if (m_dense_kernel == DenseKernel::blas)
    {
        detail::blas_apply(A, x, output);
        return;
    }
    if (A.cols() == 0)
    {
        std::ranges::fill(output, 0.0);
        return;
    }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(m_threads) if (m_threads > 1 && A.rows() >= 64 && A.cols() >= 128)
#endif
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        const ConstVectorView row{A.data() + i * A.row_stride(), A.cols()};
        output[i] = row_dot(row, x, m_fused);
    }
}

double CpuBackend::apply_and_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const
{
    detail::same_size(p, ap);
    validate_apply(A, p, ap);
    if (!m_fused || m_dense_kernel == DenseKernel::blas)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                                       \
    num_threads(m_threads) if (m_threads > 1 && A.rows() >= 64 && A.cols() >= 128)
#endif
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        const ConstVectorView row{A.data() + i * A.row_stride(), A.cols()};
        const double sum = row_dot(row, p, true);
        ap[i] = sum;
        product += p[i] * sum;
    }
    return product;
}

double CpuBackend::residual_squared_norm(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                                         ResidualSign sign) const
{
    const bool b_minus_ax = sign == ResidualSign::rhs_minus_product;
    detail::same_size(b, r);
    detail::no_overlap(b, r);
    validate_apply(A, x, r);
    if (m_dense_kernel == DenseKernel::blas)
    {
        detail::blas_apply(A, x, r);
        if (!m_fused)
        {
            axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
            return squared_norm(r);
        }
        double squared_norm = 0.0;
        for (std::size_t i = 0; i < r.size(); ++i)
        {
            const double value = b_minus_ax ? b[i] - r[i] : r[i] - b[i];
            r[i] = value;
            squared_norm += value * value;
        }
        return squared_norm;
    }
    if (A.cols() == 0)
    {
        axpby(b_minus_ax ? 1.0 : -1.0, b, 0.0, b, r);
        return squared_norm(r);
    }
    if (!m_fused)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return squared_norm(r);
    }
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                                                  \
    num_threads(m_threads) if (m_threads > 1 && A.rows() >= 64 && A.cols() >= 128)
#endif
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        const ConstVectorView row{A.data() + i * A.row_stride(), A.cols()};
        const double sum = row_dot(row, x, true);
        const double value = b_minus_ax ? b[i] - sum : sum - b[i];
        r[i] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

} // namespace cg
