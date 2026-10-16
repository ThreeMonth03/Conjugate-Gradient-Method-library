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

double row_dot(ConstVectorView row, ConstVectorView x)
{
    const auto size = row.size();
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

// Preserve the four-lane accumulation order of row_dot while sharing x loads
// between two independent rows. Padding is never part of either row view.
std::array<double, 2> row_pair_dot(ConstVectorView first, ConstVectorView second, ConstVectorView x)
{
    std::array<double, 4> first_sums{};
    std::array<double, 4> second_sums{};
    const auto size = first.size();
    const auto bulk = size - size % 4;
    std::size_t j = 0;
    for (; j < bulk; j += 4)
    {
        for (std::size_t lane = 0; lane < 4; ++lane)
        {
            const double value = x[j + lane];
            first_sums[lane] += first[j + lane] * value;
            second_sums[lane] += second[j + lane] * value;
        }
    }
    std::array<double, 2> sums{(first_sums[0] + first_sums[1]) + (first_sums[2] + first_sums[3]),
                               (second_sums[0] + second_sums[1]) + (second_sums[2] + second_sums[3])};
    for (; j < size; ++j)
    {
        const double value = x[j];
        sums[0] += first[j] * value;
        sums[1] += second[j] * value;
    }
    return sums;
}

// Keep each row's four-lane accumulation and tail order while sharing x
// between four independent rows. Callers pass only complete four-row blocks.
std::array<double, 4> row_quad_dot(DenseMatrixView A, std::size_t first_row, ConstVectorView x)
{
    const std::array<ConstVectorView, 4> rows{ConstVectorView{A.data() + first_row * A.row_stride(), A.cols()},
                                              ConstVectorView{A.data() + (first_row + 1) * A.row_stride(), A.cols()},
                                              ConstVectorView{A.data() + (first_row + 2) * A.row_stride(), A.cols()},
                                              ConstVectorView{A.data() + (first_row + 3) * A.row_stride(), A.cols()}};
    std::array<std::array<double, 4>, 4> accumulators{};
    const auto bulk = x.size() - x.size() % 4;
    std::size_t j = 0;
    for (; j < bulk; j += 4)
    {
        for (std::size_t lane = 0; lane < 4; ++lane)
        {
            const double value = x[j + lane];
            accumulators[0][lane] += rows[0][j + lane] * value;
            accumulators[1][lane] += rows[1][j + lane] * value;
            accumulators[2][lane] += rows[2][j + lane] * value;
            accumulators[3][lane] += rows[3][j + lane] * value;
        }
    }
    std::array<double, 4> sums{};
    for (std::size_t row = 0; row < 4; ++row)
    {
        sums[row] = (accumulators[row][0] + accumulators[row][1]) + (accumulators[row][2] + accumulators[row][3]);
    }
    for (; j < x.size(); ++j)
    {
        const double value = x[j];
        sums[0] += rows[0][j] * value;
        sums[1] += rows[1][j] * value;
        sums[2] += rows[2][j] * value;
        sums[3] += rows[3][j] * value;
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
    const auto blocked_rows = A.rows() - A.rows() % 4;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                                                              \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < blocked_rows; i += 4)
    {
        const auto sums = row_quad_dot(A, i, x);
        for (std::size_t row = 0; row < 4; ++row)
        {
            output[i + row] = sums[row];
        }
    }
    auto tail = blocked_rows;
    if (A.rows() - tail >= 2)
    {
        const ConstVectorView first{A.data() + tail * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (tail + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, x);
        output[tail] = sums[0];
        output[tail + 1] = sums[1];
        tail += 2;
    }
    if (tail != A.rows())
    {
        const ConstVectorView row{A.data() + tail * A.row_stride(), A.cols()};
        output[tail] = row_dot(row, x);
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
    const auto blocked_rows = A.rows() - A.rows() % 4;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                                       \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < blocked_rows; i += 4)
    {
        const auto sums = row_quad_dot(A, i, p);
        for (std::size_t pair = 0; pair < 4; pair += 2)
        {
            ap[i + pair] = sums[pair];
            ap[i + pair + 1] = sums[pair + 1];
            product += p[i + pair] * sums[pair];
            product += p[i + pair + 1] * sums[pair + 1];
        }
    }
    auto tail = blocked_rows;
    if (A.rows() - tail >= 2)
    {
        const ConstVectorView first{A.data() + tail * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (tail + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, p);
        ap[tail] = sums[0];
        ap[tail + 1] = sums[1];
        product += p[tail] * sums[0];
        product += p[tail + 1] * sums[1];
        tail += 2;
    }
    if (tail != A.rows())
    {
        const ConstVectorView row{A.data() + tail * A.row_stride(), A.cols()};
        const double sum = row_dot(row, p);
        ap[tail] = sum;
        product += p[tail] * sum;
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
    const auto blocked_rows = A.rows() - A.rows() % 4;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                                                  \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= detail::openmp_min_dense_rows &&                   \
                                       A.cols() >= detail::openmp_min_dense_columns)
#endif
    for (std::size_t i = 0; i < blocked_rows; i += 4)
    {
        const auto sums = row_quad_dot(A, i, x);
        for (std::size_t pair = 0; pair < 4; pair += 2)
        {
            const double first_value = b_minus_ax ? b[i + pair] - sums[pair] : sums[pair] - b[i + pair];
            const double second_value =
                b_minus_ax ? b[i + pair + 1] - sums[pair + 1] : sums[pair + 1] - b[i + pair + 1];
            r[i + pair] = first_value;
            r[i + pair + 1] = second_value;
            squared_norm += first_value * first_value;
            squared_norm += second_value * second_value;
        }
    }
    auto tail = blocked_rows;
    if (A.rows() - tail >= 2)
    {
        const ConstVectorView first{A.data() + tail * A.row_stride(), A.cols()};
        const ConstVectorView second{A.data() + (tail + 1) * A.row_stride(), A.cols()};
        const auto sums = row_pair_dot(first, second, x);
        const double first_value = b_minus_ax ? b[tail] - sums[0] : sums[0] - b[tail];
        const double second_value = b_minus_ax ? b[tail + 1] - sums[1] : sums[1] - b[tail + 1];
        r[tail] = first_value;
        r[tail + 1] = second_value;
        squared_norm += first_value * first_value;
        squared_norm += second_value * second_value;
        tail += 2;
    }
    if (tail != A.rows())
    {
        const ConstVectorView row{A.data() + tail * A.row_stride(), A.cols()};
        const double sum = row_dot(row, x);
        const double value = b_minus_ax ? b[tail] - sum : sum - b[tail];
        r[tail] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

} // namespace cg
