#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/detail/blas.hpp>
#include <cg/backend/cpu/detail/parallel_thresholds.hpp>
#include <cg/backend/cpu/detail/validation.hpp>
#include <cg/matrix/detail/validation.hpp>

#include <algorithm>
#include <array>

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

double reduce_row(const std::array<double, 8> &sums)
{
    return ((sums[0] + sums[1]) + (sums[2] + sums[3])) + ((sums[4] + sums[5]) + (sums[6] + sums[7]));
}

double row_dot(ConstVectorView row, ConstVectorView x)
{
    const auto size = row.size();
    // Eight independent chains shorten the dependency within each lane. The
    // balanced reduction changes rounding, without fast-math or a wider ISA.
    std::array<double, 8> sums{};
    const auto bulk = size - size % 8;
    std::size_t j = 0;
    for (; j < bulk; j += 8)
    {
        for (std::size_t lane = 0; lane < 8; ++lane)
        {
            sums[lane] += row[j + lane] * x[j + lane];
        }
    }
    if (size - j >= 4)
    {
        for (std::size_t lane = 0; lane < 4; ++lane)
        {
            sums[lane] += row[j + lane] * x[j + lane];
        }
        j += 4;
    }
    double sum = reduce_row(sums);
    for (; j < size; ++j)
    {
        sum += row[j] * x[j];
    }
    return sum;
}

// Use the same eight-lane accumulation as row_dot while sharing x loads
// between two independent rows. Padding is never part of either row view.
std::array<double, 2> row_pair_dot(ConstVectorView first, ConstVectorView second, ConstVectorView x)
{
    std::array<double, 8> first_sums{};
    std::array<double, 8> second_sums{};
    const auto size = first.size();
    const auto bulk = size - size % 8;
    std::size_t j = 0;
    for (; j < bulk; j += 8)
    {
        for (std::size_t lane = 0; lane < 8; ++lane)
        {
            const double value = x[j + lane];
            first_sums[lane] += first[j + lane] * value;
            second_sums[lane] += second[j + lane] * value;
        }
    }
    if (size - j >= 4)
    {
        for (std::size_t lane = 0; lane < 4; ++lane)
        {
            const double value = x[j + lane];
            first_sums[lane] += first[j + lane] * value;
            second_sums[lane] += second[j + lane] * value;
        }
        j += 4;
    }
    std::array<double, 2> sums{reduce_row(first_sums), reduce_row(second_sums)};
    for (; j < size; ++j)
    {
        const double value = x[j];
        sums[0] += first[j] * value;
        sums[1] += second[j] * value;
    }
    return sums;
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
    const auto paired_rows = A.rows() - A.rows() % 2;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                                                              \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < paired_rows; i += 2)
    {
        const ConstVectorView first{A.data() + i * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (i + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, x);
        output[i] = sums[0];
        output[i + 1] = sums[1];
    }
    if (paired_rows != A.rows())
    {
        const ConstVectorView row{A.data() + paired_rows * A.row_stride(), A.cols()};
        output[paired_rows] = row_dot(row, x);
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
    const auto paired_rows = A.rows() - A.rows() % 2;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                                       \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < paired_rows; i += 2)
    {
        const ConstVectorView first{A.data() + i * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (i + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, p);
        ap[i] = sums[0];
        ap[i + 1] = sums[1];
        product += p[i] * sums[0];
        product += p[i + 1] * sums[1];
    }
    if (paired_rows != A.rows())
    {
        const ConstVectorView row{A.data() + paired_rows * A.row_stride(), A.cols()};
        const double sum = row_dot(row, p);
        ap[paired_rows] = sum;
        product += p[paired_rows] * sum;
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
    const auto paired_rows = A.rows() - A.rows() % 2;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                                                  \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < paired_rows; i += 2)
    {
        const ConstVectorView first{A.data() + i * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (i + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, x);
        const double first_value = b_minus_ax ? b[i] - sums[0] : sums[0] - b[i];
        const double second_value = b_minus_ax ? b[i + 1] - sums[1] : sums[1] - b[i + 1];
        r[i] = first_value;
        r[i + 1] = second_value;
        squared_norm += first_value * first_value;
        squared_norm += second_value * second_value;
    }
    if (paired_rows != A.rows())
    {
        const ConstVectorView row{A.data() + paired_rows * A.row_stride(), A.cols()};
        const double sum = row_dot(row, x);
        const double value = b_minus_ax ? b[paired_rows] - sum : sum - b[paired_rows];
        r[paired_rows] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

} // namespace cg
