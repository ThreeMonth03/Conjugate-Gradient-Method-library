#include <cg/cg.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

template <typename Matrix>
constexpr bool has_indptr = requires(Matrix &&matrix) { std::forward<Matrix>(matrix).indptr(); };
template <typename Matrix>
constexpr bool has_indices = requires(Matrix &&matrix) { std::forward<Matrix>(matrix).indices(); };
template <typename Matrix>
constexpr bool has_values = requires(Matrix &&matrix) { std::forward<Matrix>(matrix).values(); };

static_assert(has_indptr<cg::CsrMatrix &> && has_indptr<const cg::CsrMatrix &>);
static_assert(has_indices<cg::CsrMatrix &> && has_indices<const cg::CsrMatrix &>);
static_assert(has_values<cg::CsrMatrix &> && has_values<const cg::CsrMatrix &>);
static_assert(!has_indptr<cg::CsrMatrix> && !has_indptr<const cg::CsrMatrix>);
static_assert(!has_indices<cg::CsrMatrix> && !has_indices<const cg::CsrMatrix>);
static_assert(!has_values<cg::CsrMatrix> && !has_values<const cg::CsrMatrix>);
static_assert(std::is_same_v<decltype(std::declval<cg::CsrMatrix &>().indptr()), std::span<const std::size_t>>);
static_assert(std::is_same_v<decltype(std::declval<const cg::CsrMatrix &>().indptr()), std::span<const std::size_t>>);
static_assert(std::is_same_v<decltype(std::declval<cg::CsrMatrix &>().indices()), std::span<const std::size_t>>);
static_assert(std::is_same_v<decltype(std::declval<const cg::CsrMatrix &>().indices()), std::span<const std::size_t>>);
static_assert(std::is_same_v<decltype(std::declval<cg::CsrMatrix &>().values()), cg::ConstVectorView>);
static_assert(std::is_same_v<decltype(std::declval<const cg::CsrMatrix &>().values()), cg::ConstVectorView>);
static_assert(noexcept(std::declval<const cg::CsrMatrix &>().indptr()));
static_assert(noexcept(std::declval<const cg::CsrMatrix &>().indices()));
static_assert(noexcept(std::declval<const cg::CsrMatrix &>().values()));

cg::CsrMatrix poisson(std::size_t width, std::size_t height)
{
    const auto size = width * height;
    std::vector<std::size_t> indptr{0}, indices;
    std::vector<double> values;
    for (std::size_t row = 0; row < size; ++row)
    {
        if (row >= width)
        {
            indices.push_back(row - width);
            values.push_back(-1);
        }
        if (row % width != 0)
        {
            indices.push_back(row - 1);
            values.push_back(-1);
        }
        indices.push_back(row);
        values.push_back(4);
        if (row % width + 1 != width)
        {
            indices.push_back(row + 1);
            values.push_back(-1);
        }
        if (row + width < size)
        {
            indices.push_back(row + width);
            values.push_back(-1);
        }
        indptr.push_back(indices.size());
    }
    return {size, size, std::move(indptr), std::move(indices), std::move(values)};
}

TEST(CSR, ViewsBorrowReadOnlyCanonicalStorage)
{
    std::vector<std::size_t> indptr{0, 1, 1, 2}, indices{0, 2};
    std::vector<double> values{2, 3};
    const auto *indptr_storage = indptr.data();
    const auto *indices_storage = indices.data();
    const auto *values_storage = values.data();
    cg::CsrMatrix A(3, 3, std::move(indptr), std::move(indices), std::move(values));
    EXPECT_EQ(A.indptr().data(), indptr_storage);
    EXPECT_EQ(A.indices().data(), indices_storage);
    EXPECT_EQ(A.values().data(), values_storage);
    EXPECT_EQ(A.indptr().size(), A.rows() + 1);
    EXPECT_EQ(A.indices().size(), A.nnz());
    EXPECT_EQ(A.values().size(), A.nnz());
    const auto &read_only = A;
    EXPECT_EQ(read_only.indptr().data(), A.indptr().data());
    EXPECT_EQ(read_only.indices().data(), A.indices().data());
    EXPECT_EQ(read_only.values().data(), A.values().data());
    EXPECT_TRUE(std::ranges::equal(A.indptr(), std::vector<std::size_t>{0, 1, 1, 2}));
    EXPECT_TRUE(std::ranges::equal(A.indices(), std::vector<std::size_t>{0, 2}));
    EXPECT_TRUE(std::ranges::equal(A.values(), std::vector<double>{2, 3}));
}

TEST(CSR, OwnsCanonicalArraysAndSupportsEmptyRows)
{
    const cg::CsrMatrix A(3, 3, {0, 1, 1, 2}, {0, 2}, {2, 3});
    EXPECT_EQ(A.rows(), 3u);
    EXPECT_EQ(A.cols(), 3u);
    EXPECT_EQ(A.nnz(), 2u);
    std::vector<double> x({1, 2, 3}), y(3);
    const cg::CpuBackend backend;
    backend.apply(A, x, y);
    EXPECT_DOUBLE_EQ(y.data()[0], 2);
    EXPECT_DOUBLE_EQ(y.data()[1], 0);
    EXPECT_DOUBLE_EQ(y.data()[2], 9);
    const cg::CsrMatrix empty(0, 0, {0}, {}, {});
    EXPECT_EQ(empty.nnz(), 0u);
    backend.apply(empty, {}, {});
}

TEST(CSR, RejectsMalformedPointersDimensionsColumnsAndDuplicates)
{
    EXPECT_THROW((cg::CsrMatrix(2, 2, {0, 1}, {0}, {1})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(1, 1, {1, 1}, {}, {})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(1, 1, {0, 0}, {0}, {})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(2, 2, {0, 2, 1}, {0}, {1})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(1, 2, {0, 2}, {0, 2}, {1, 1})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(1, 2, {0, 2}, {1, 0}, {1, 1})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(1, 2, {0, 2}, {0, 0}, {1, 1})), std::invalid_argument);
    EXPECT_THROW((cg::CsrMatrix(std::numeric_limits<std::size_t>::max(), 1, {}, {}, {})), std::invalid_argument);
}

TEST(CSR, MovedFromOwnersCannotRetainAStaleShape)
{
    cg::CsrMatrix source(2, 2, {0, 1, 2}, {0, 1}, {2, 3});
    cg::CsrMatrix destination(std::move(source));
    EXPECT_EQ(destination.rows(), 2u);
    EXPECT_EQ(source.rows(), 0u);
    const cg::CpuBackend backend;
    backend.apply(source, {}, {});
    source = std::move(destination);
    EXPECT_EQ(source.rows(), 2u);
    EXPECT_EQ(destination.rows(), 0u);
    backend.apply(destination, {}, {});
    const auto *storage = source.values().data();
    auto &alias = source;
    source = std::move(alias);
    EXPECT_EQ(source.rows(), 2u);
    EXPECT_EQ(source.values().data(), storage);
    EXPECT_TRUE(std::ranges::equal(source.values(), std::vector<double>{2, 3}));
}

TEST(CSR, RectangularApplyIsAllowedButCGAndApplyDotRequireSquare)
{
    const cg::CsrMatrix A(2, 3, {0, 2, 3}, {0, 2, 1}, {2, 3, 4});
    std::vector<double> x({1, 2, 3}), y(2), b(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    backend.apply(A, x, y);
    EXPECT_DOUBLE_EQ(y.data()[0], 11);
    EXPECT_DOUBLE_EQ(y.data()[1], 8);
    EXPECT_THROW(backend.apply_and_dot(A, x, y), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(cg::solve_into(A, b, y, workspace, backend)), std::invalid_argument);
}

TEST(CSR, RejectsOverlappingOutputs)
{
    const cg::CsrMatrix A(2, 2, {0, 1, 2}, {0, 1}, {2, 3});
    std::vector<double> x({1, 2}), b({3, 4}), r(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    EXPECT_THROW(backend.apply(A, x, x), std::invalid_argument);
    EXPECT_THROW(backend.residual_squared_norm(A, b, x, b), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(cg::solve_into(A, b, b, workspace, backend)), std::invalid_argument);
    // No write occurs: reject a view of the owner's coefficient storage first.
    cg::VectorView coefficients{const_cast<double *>(A.values().data()), A.nnz()};
    EXPECT_THROW(backend.apply(A, x, coefficients), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(cg::solve_into(A, b, coefficients, workspace, backend)), std::invalid_argument);
}

TEST(CSR, FusedOperationsMatchReferenceWithParallelRows)
{
    const auto A = poisson(65, 65);
    std::vector<double> p(A.rows()), b(A.rows()), ap(A.rows()), reference_ap(A.rows());
    std::vector<double> r(A.rows()), reference_r(A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        p.data()[i] = std::sin(static_cast<double>(i));
        b.data()[i] = std::cos(static_cast<double>(i));
    }
    const cg::CpuBackend reference(1, false);
    const int threads = cg::CpuBackend::openmp_enabled() ? 2 : 1;
    const cg::CpuBackend fused(threads, true);
    const double expected_dot = reference.apply_and_dot(A, p, reference_ap);
    EXPECT_NEAR(fused.apply_and_dot(A, p, ap), expected_dot, std::abs(expected_dot) * 1e-12);
    const double expected_norm = reference.residual_squared_norm(A, b, p, reference_r);
    EXPECT_NEAR(fused.residual_squared_norm(A, b, p, r), expected_norm, expected_norm * 1e-12);
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        EXPECT_DOUBLE_EQ(ap.data()[i], reference_ap.data()[i]);
        EXPECT_DOUBLE_EQ(r.data()[i], reference_r.data()[i]);
    }
}

TEST(CSR, BothAlgorithmsSolveTheSamePoissonProblemAndReportTrueResidual)
{
    const auto A = poisson(9, 7);
    std::vector<double> truth(A.rows()), b(A.rows());
    const cg::CpuBackend backend;
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        truth.data()[i] = 0.3 + std::sin(static_cast<double>(i));
    }
    backend.apply(A, truth, b);
    for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
    {
        std::vector<double> x(A.rows()), check(A.rows());
        cg::Workspace workspace(A.rows());
        cg::CgOptions options;
        options.algorithm = algorithm;
        options.rtol = 1e-10;
        const auto info = cg::solve_into(A, b, x, workspace, backend, options);
        EXPECT_EQ(info.status, cg::Status::converged);
        EXPECT_LE(info.residual_norm, options.rtol * std::sqrt(backend.squared_norm(b)));
        EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(backend.residual_squared_norm(A, b, x, check)));
        for (std::size_t i = 0; i < A.rows(); ++i)
        {
            EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-8);
        }
    }
}

TEST(CSR, ParallelSolveMatchesSerialTrueResidualWithinTolerance)
{
    if (!cg::CpuBackend::openmp_enabled())
    {
        GTEST_SKIP() << "OpenMP is unavailable";
    }
    const auto A = poisson(129, 129);
    std::vector<double> b(A.rows()), serial_x(A.rows()), parallel_x(A.rows()), check(A.rows());
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        b.data()[i] = std::sin(static_cast<double>(i));
    }
    cg::Workspace serial_workspace(A.rows()), parallel_workspace(A.rows());
    const cg::CpuBackend serial(1), parallel(2);
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.maxiter = 3;
    const auto first = cg::solve_into(A, b, serial_x, serial_workspace, serial, options);
    const auto second = cg::solve_into(A, b, parallel_x, parallel_workspace, parallel, options);
    EXPECT_EQ(first.status, cg::Status::max_iterations);
    EXPECT_EQ(second.status, cg::Status::max_iterations);
    EXPECT_EQ(first.matvecs, second.matvecs);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_NEAR(first.residual_norm, second.residual_norm, first.residual_norm * 1e-12);
    for (std::size_t i = 0; i < A.rows(); ++i)
    {
        EXPECT_NEAR(serial_x.data()[i], parallel_x.data()[i], 1e-12);
    }
}

TEST(CSR, NonfiniteCoefficientsProduceAStatus)
{
    const cg::CsrMatrix A(1, 1, {0, 1}, {0}, {std::numeric_limits<double>::infinity()});
    std::vector<double> b(std::vector<double>{1}), x(1);
    cg::Workspace workspace(1);
    const cg::CpuBackend backend;
    const auto info = cg::solve_into(A, b, x, workspace, backend);
    EXPECT_EQ(info.status, cg::Status::non_finite);
}

} // namespace
