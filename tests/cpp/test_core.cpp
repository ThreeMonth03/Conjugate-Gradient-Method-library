#include "cg/cg.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <vector>

namespace
{

using cg::ConstVectorView;
using cg::VectorView;

cg::DenseMatrix tridiagonal(std::size_t n)
{
    cg::DenseMatrix A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        A.data()[i * n + i] = 2.0;
        if (i != 0)
        {
            A.data()[i * n + i - 1] = -1.0;
        }
        if (i + 1 != n)
        {
            A.data()[i * n + i + 1] = -1.0;
        }
    }
    return A;
}

TEST(Storage, OwnsItsValuesAndHasConstViews)
{
    std::vector<double> source{1.0, 2.0, 3.0};
    cg::Vector owned(source);
    source[0] = 100.0;
    EXPECT_EQ(owned.data()[0], 1.0);
    const auto &immutable = owned;
    ConstVectorView view = immutable.view();
    EXPECT_EQ(view.size, 3u);
    EXPECT_EQ(view.data[2], 3.0);
}

TEST(Storage, ChecksShapeAndOverflowBeforeAccess)
{
    EXPECT_THROW((cg::DenseMatrix(2, 3, {1.0})), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(nullptr, 2, 2)), std::invalid_argument);
    double value = 0.0;
    EXPECT_THROW((cg::DenseMatrixView(&value, 2, 3, 2)), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(&value, std::numeric_limits<std::size_t>::max(), 2)),
                 std::invalid_argument);
}

TEST(Storage, RejectsUnalignedViewsBeforeReadingTheirData)
{
    alignas(double) unsigned char storage[sizeof(double) * 4 + 1] = {};
    const auto *unaligned = reinterpret_cast<const double *>(storage + 1);
    const cg::CpuBackend backend;
    EXPECT_THROW((cg::DenseMatrixView(unaligned, 2, 2)), std::invalid_argument);
    EXPECT_THROW(backend.dot({unaligned, 1}, {unaligned, 1}), std::invalid_argument);
}

TEST(Workspace, ReusesStorageAcrossSameAndSmallerSizes)
{
    cg::Workspace workspace(32);
    const auto *pointer = workspace.residual().data;
    const auto capacity = workspace.capacity();
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data, pointer);
    workspace.prepare(5);
    EXPECT_EQ(workspace.residual().data, pointer);
    EXPECT_EQ(workspace.capacity(), capacity);
    EXPECT_EQ(workspace.size(), 5u);
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data, pointer);
    workspace.prepare(64);
    EXPECT_EQ(workspace.size(), 64u);
    EXPECT_GE(workspace.capacity(), 64u);
}

TEST(Backend, ValidatesThreadCountAndKeepsContextLocal)
{
    EXPECT_THROW(cg::CpuBackend(0), std::invalid_argument);
    EXPECT_THROW(cg::CpuBackend(-1), std::invalid_argument);
    const cg::CpuBackend first(1, false);
    const cg::CpuBackend second(1, true);
    EXPECT_FALSE(first.fused());
    EXPECT_TRUE(second.fused());
    if (!cg::CpuBackend::openmp_enabled())
    {
        EXPECT_THROW(cg::CpuBackend(2), std::invalid_argument);
    }
    else
    {
        const cg::CpuBackend parallel(2);
        EXPECT_EQ(first.threads(), 1);
        EXPECT_EQ(parallel.threads(), 2);
    }
}

TEST(Backend, HandlesPaddedRowMajorStorage)
{
    const std::vector<double> values{1, 2, 99, 3, 4, 99};
    const cg::DenseMatrixView A(values.data(), 2, 2, 3);
    const cg::Vector input({2, 3});
    cg::Vector output(2);
    const cg::CpuBackend backend;
    backend.apply(A, input.view(), output.view());
    EXPECT_DOUBLE_EQ(output.data()[0], 8);
    EXPECT_DOUBLE_EQ(output.data()[1], 18);
    EXPECT_DOUBLE_EQ(backend.apply_dot(A, input.view(), output.view()), 70);
}

TEST(Backend, EmptyColumnsProduceZeroProductsWithoutDereferencingNullStorage)
{
    const cg::DenseMatrixView A(nullptr, 3, 0);
    const cg::Vector b({1, 2, 3});
    cg::Vector output({10, 20, 30});
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused);
        backend.apply(A, {}, output.view());
        for (std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(output.data()[i], 0);
        }
        EXPECT_DOUBLE_EQ(backend.residual(A, b.view(), {}, output.view()), 14);
        EXPECT_DOUBLE_EQ(output.data()[1], 2);
        EXPECT_DOUBLE_EQ(backend.residual(A, b.view(), {}, output.view(), false), 14);
        EXPECT_DOUBLE_EQ(output.data()[1], -2);
    }
}

TEST(Backend, SupportsExactInPlaceAxpbyAndRejectsPartialOverlap)
{
    cg::Vector x({1, 2, 3});
    cg::Vector y({4, 5, 6});
    const cg::CpuBackend backend;
    backend.axpby(2.0, x.view(), 3.0, y.view(), y.view());
    EXPECT_DOUBLE_EQ(y.data()[0], 14);
    EXPECT_DOUBLE_EQ(y.data()[2], 24);
    backend.axpy(2.0, x.view(), x.view());
    EXPECT_DOUBLE_EQ(x.data()[2], 9);
    EXPECT_THROW(backend.copy({x.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
    EXPECT_THROW(backend.axpy(1.0, {x.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
    EXPECT_THROW(backend.axpby(1.0, {x.data(), 2}, 1.0, {y.data(), 2}, {x.data() + 1, 2}),
                 std::invalid_argument);
}

TEST(Backend, RejectsDimensionsNullPointersAndMatvecAliasing)
{
    cg::DenseMatrix A(2, 2, {1, 0, 0, 1});
    cg::Vector x({1, 2});
    cg::Vector y(2);
    const cg::CpuBackend backend;
    EXPECT_THROW(backend.dot(x.view(), {y.data(), 1}), std::invalid_argument);
    EXPECT_THROW(backend.dot({nullptr, 2}, x.view()), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x.view(), x.view()), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x.view(), {A.data(), 2}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(backend.dot({}, {}), 0.0);
}

TEST(Backend, ResidualSignIsExplicit)
{
    cg::DenseMatrix A(2, 2, {2, 0, 0, 3});
    cg::Vector b({5, 9});
    cg::Vector x({1, 2});
    cg::Vector r(2);
    const cg::CpuBackend backend;
    EXPECT_DOUBLE_EQ(backend.residual(A.view(), b.view(), x.view(), r.view()), 18);
    EXPECT_DOUBLE_EQ(r.data()[0], 3);
    EXPECT_DOUBLE_EQ(r.data()[1], 3);
    EXPECT_DOUBLE_EQ(backend.residual(A.view(), b.view(), x.view(), r.view(), false), 18);
    EXPECT_DOUBLE_EQ(r.data()[0], -3);
}

TEST(Backend, StableNormHandlesTinyLargeAndSubnormalValues)
{
    const cg::CpuBackend backend;
    for (const double magnitude : {1e-200, 1e-160, 1e200, std::numeric_limits<double>::denorm_min(),
                                   std::numeric_limits<double>::max() / 4.0})
    {
        const cg::Vector values({magnitude, -magnitude});
        const double expected = std::hypot(magnitude, magnitude);
        const double ulp =
            std::nextafter(expected, std::numeric_limits<double>::infinity()) - expected;
        EXPECT_NEAR(backend.norm(values.view()), expected, 8.0 * ulp);
    }
    const cg::Vector zero(3);
    EXPECT_DOUBLE_EQ(backend.norm(zero.view()), 0);
    EXPECT_DOUBLE_EQ(backend.norm({}), 0);
    const cg::Vector infinite({std::numeric_limits<double>::infinity(), 1.0});
    const cg::Vector nan({std::numeric_limits<double>::quiet_NaN(), 1.0});
    EXPECT_TRUE(std::isinf(backend.norm(infinite.view())));
    EXPECT_TRUE(std::isnan(backend.norm(nan.view())));
}

TEST(CGRange, DoesNotClaimConvergenceWhenSquaredResidualUnderflowsOrOverflows)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::CpuBackend backend;
    for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
    {
        for (const double magnitude : {1e-200, 1e-160, 1e200})
        {
            const cg::Vector b({magnitude, -magnitude});
            cg::Vector x(2);
            cg::Workspace workspace(2);
            cg::CgOptions options;
            options.algorithm = algorithm;
            const auto info =
                cg::solve_into(identity.view(), b.view(), x.view(), workspace, backend, options);
            EXPECT_EQ(info.status, cg::Status::non_finite);
            EXPECT_EQ(info.iterations, 0u);
            EXPECT_EQ(info.matvecs, 1u);
            EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
            EXPECT_GT(info.residual_norm, options.rtol * std::hypot(magnitude, magnitude));
            EXPECT_DOUBLE_EQ(x.data()[0], 0);
        }
    }
}

TEST(CGRange, AcceptsActualConvergenceAndUsableScaling)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::CpuBackend backend;
    for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
    {
        cg::CgOptions options;
        options.algorithm = algorithm;
        for (const double magnitude : {1e-150, 1e150})
        {
            const cg::Vector b({magnitude, -magnitude});
            cg::Vector x(2);
            cg::Workspace workspace(2);
            const auto info =
                cg::solve_into(identity.view(), b.view(), x.view(), workspace, backend, options);
            EXPECT_EQ(info.status, cg::Status::converged);
            EXPECT_EQ(info.iterations, 1u);
            EXPECT_DOUBLE_EQ(info.residual_norm, 0);
        }
        const cg::Vector tiny({1e-200, -1e-200});
        cg::Vector x(2);
        cg::Workspace workspace(2);
        options.atol = 1e-199;
        const auto accepted =
            cg::solve_into(identity.view(), tiny.view(), x.view(), workspace, backend, options);
        EXPECT_EQ(accepted.status, cg::Status::converged);
        EXPECT_EQ(accepted.iterations, 0u);
        EXPECT_GT(accepted.residual_norm, 0);
        EXPECT_LE(accepted.residual_norm, options.atol);
        options.atol = 0;
        options.maxiter = 0;
        const auto limited =
            cg::solve_into(identity.view(), tiny.view(), x.view(), workspace, backend, options);
        EXPECT_EQ(limited.status, cg::Status::max_iterations);
        EXPECT_GT(limited.residual_norm, 0);
    }
}

TEST(CGRange, ZeroOrSubnormalCurvatureIsReportedAsNumericalRangeFailure)
{
    const cg::CpuBackend backend;
    for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
    {
        for (const double diagonal : {1e-310, 1e-200})
        {
            const cg::DenseMatrix A(1, 1, {diagonal});
            const double magnitude = diagonal == 1e-310 ? 1.0 : 1e-150;
            const cg::Vector b(std::vector<double>{magnitude});
            cg::Vector x(1);
            cg::Workspace workspace(1);
            cg::CgOptions options;
            options.algorithm = algorithm;
            const auto info =
                cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
            EXPECT_EQ(info.status, cg::Status::non_finite);
            EXPECT_EQ(info.iterations, 0u);
            EXPECT_DOUBLE_EQ(info.residual_norm, magnitude);
        }
    }
}

TEST(CGExplicit, SolvesFromAnIndependentInitialGuess)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({1, 2});
    cg::Vector x({0.5, -0.25});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.rtol = 1e-12;
    const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_GT(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1 + 2 * info.iterations);
    EXPECT_NEAR(x.data()[0], 1.0 / 11, 1e-12);
    EXPECT_NEAR(x.data()[1], 7.0 / 11, 1e-12);
    EXPECT_LE(info.residual_norm, options.rtol * std::sqrt(5.0));
}

TEST(CGExplicit, ChecksInitialResidualAndZeroRightHandSide)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({5, 4});
    cg::Vector x({1, 1});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
}

TEST(CGExplicit, ZeroRightHandSideConvergesWithoutAnIteration)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b(2);
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.residual_norm, 0);
}

TEST(CGExplicit, ReportsIterationLimitAndTrueResidual)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({1, 2});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.maxiter = 0;
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(5.0));
    options.maxiter = 1;
    info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 1u);
    cg::Vector check(2);
    EXPECT_DOUBLE_EQ(info.residual_norm,
                     std::sqrt(backend.residual(A.view(), b.view(), x.view(), check.view())));
}

TEST(CGExplicit, ReportsBreakdownAndNonfiniteData)
{
    cg::DenseMatrix indefinite(2, 2, {-1, 0, 0, -1});
    cg::Vector b({1, 2});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    auto info = cg::solve_into(indefinite.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 2u);
    indefinite.data()[0] = std::numeric_limits<double>::quiet_NaN();
    info = cg::solve_into(indefinite.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::non_finite);
}

TEST(CGExplicit, ValidatesOptionsStorageAndWorkspace)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({1, 2});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.rtol = -1;
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options),
                 std::invalid_argument);
    options.rtol = std::numeric_limits<double>::infinity();
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options),
                 std::invalid_argument);
    options.rtol = 1e-8;
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), b.view(), workspace, backend, options),
                 std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), {A.data(), 2}, workspace, backend),
                 std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), workspace.residual(), workspace, backend),
                 std::invalid_argument);
    workspace.prepare(1);
    EXPECT_THROW(cg::solve_into(A.view(), b.view(), x.view(), workspace, backend),
                 std::invalid_argument);
}

TEST(CGExplicit, ReusingWorkspaceResetsAllIterationState)
{
    const auto A = tridiagonal(31);
    cg::Vector truth(31);
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        truth.data()[i] = 0.25 + std::sin(static_cast<double>(i));
    }
    cg::Vector b(31);
    cg::Vector x(31);
    cg::Workspace workspace(31);
    const cg::CpuBackend backend;
    backend.apply(A.view(), truth.view(), b.view());
    const auto *pointer = workspace.residual().data;
    const auto first = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(first.status, cg::Status::converged);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-6);
        x.data()[i] = 0;
    }
    const auto second = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(second.status, cg::Status::converged);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_EQ(workspace.residual().data, pointer);
}

TEST(FusedOperations, MatchesReferenceVectorUpdatesIncludingSimdTails)
{
    for (const std::size_t size : {1u, 3u, 17u, 127u, 16385u})
    {
        cg::Vector x(size), r(size), p(size), ap(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            x.data()[i] = std::sin(static_cast<double>(i));
            r.data()[i] = std::cos(static_cast<double>(i));
            p.data()[i] = 0.1 + static_cast<double>(i % 7);
            ap.data()[i] = -0.2 + static_cast<double>(i % 11);
        }
        cg::Vector reference_x = x;
        cg::Vector reference_r = r;
        const cg::CpuBackend fused(1, true);
        const cg::CpuBackend reference(1, false);
        const auto actual = fused.update_x_r_norm2(x.view(), r.view(), p.view(), ap.view(), 0.25);
        const auto expected = reference.update_x_r_norm2(reference_x.view(), reference_r.view(),
                                                         p.view(), ap.view(), 0.25);
        EXPECT_NEAR(actual, expected, std::max(1.0, expected) * 1e-12);
        for (std::size_t i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(x.data()[i], reference_x.data()[i]);
            EXPECT_DOUBLE_EQ(r.data()[i], reference_r.data()[i]);
        }
        EXPECT_THROW(fused.update_x_r_norm2(x.view(), x.view(), p.view(), ap.view(), 0.25),
                     std::invalid_argument);
    }
}

TEST(FusedOperations, MatchesReferenceDenseReductionAndResidual)
{
    for (const std::size_t size : {1u, 3u, 17u, 129u})
    {
        const auto A = tridiagonal(size);
        cg::Vector p(size), b(size), ap(size), reference_ap(size), r(size), reference_r(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            p.data()[i] = std::sin(static_cast<double>(i)) + 0.5;
            b.data()[i] = std::cos(static_cast<double>(i));
        }
        const int threads = cg::CpuBackend::openmp_enabled() ? 2 : 1;
        const cg::CpuBackend fused(threads, true);
        const cg::CpuBackend reference(threads, false);
        EXPECT_NEAR(fused.apply_dot(A.view(), p.view(), ap.view()),
                    reference.apply_dot(A.view(), p.view(), reference_ap.view()), 1e-10);
        EXPECT_NEAR(fused.residual(A.view(), b.view(), p.view(), r.view()),
                    reference.residual(A.view(), b.view(), p.view(), reference_r.view()), 1e-10);
        for (std::size_t i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(ap.data()[i], reference_ap.data()[i]);
            EXPECT_DOUBLE_EQ(r.data()[i], reference_r.data()[i]);
        }
    }
}

TEST(CGRecurrence, SolvesAndVerifiesTrueResidualWithBothKernelPolicies)
{
    const auto A = tridiagonal(31);
    cg::Vector truth(31), b(31);
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        truth.data()[i] = 0.25 + std::sin(static_cast<double>(i));
    }
    const cg::CpuBackend reference;
    reference.apply(A.view(), truth.view(), b.view());
    for (const bool fused : {false, true})
    {
        cg::Vector x(31), check(31);
        cg::Workspace workspace(31);
        const cg::CpuBackend backend(1, fused);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        options.residual_refresh = 0;
        options.rtol = 1e-10;
        const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
        EXPECT_EQ(info.status, cg::Status::converged);
        EXPECT_EQ(info.restarts, 0u);
        EXPECT_EQ(info.matvecs, 2 + info.iterations);
        EXPECT_DOUBLE_EQ(info.residual_norm,
                         std::sqrt(backend.residual(A.view(), b.view(), x.view(), check.view())));
        EXPECT_LE(info.residual_norm, options.rtol * std::sqrt(backend.norm2(b.view())));
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-7);
        }
    }
}

TEST(CGRecurrence, CountsRefreshMatvecsAndRestartsDirections)
{
    cg::DenseMatrix A(3, 3, {2, 0, 0, 0, 3, 0, 0, 0, 4});
    cg::Vector b({1, 2, 3});
    cg::Vector x(3), check(3);
    cg::Workspace workspace(3);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.rtol = 0;
    options.atol = 0;
    options.residual_refresh = 1;
    options.maxiter = 3;
    const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 3u);
    EXPECT_EQ(info.restarts, 2u);
    EXPECT_EQ(info.matvecs, 7u);
    EXPECT_DOUBLE_EQ(info.residual_norm,
                     std::sqrt(backend.residual(A.view(), b.view(), x.view(), check.view())));
}

TEST(CGRecurrence, ChecksInitialGuessIterationLimitAndBreakdown)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({5, 4}), x({1, 1});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
    x.data()[0] = x.data()[1] = 0;
    options.maxiter = 0;
    info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.matvecs, 1u);
    options.maxiter = 1;
    info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.matvecs, 3u);
    cg::DenseMatrix indefinite(2, 2, {-1, 0, 0, -1});
    x.data()[0] = x.data()[1] = 0;
    info = cg::solve_into(indefinite.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 3u);
}

TEST(CGRecurrence, DistinctWorkspacesCanSolveOnOneContextConcurrently)
{
    const auto A = tridiagonal(31);
    cg::Vector b(31);
    b.data()[0] = b.data()[30] = 1.0;
    const cg::CpuBackend backend;
    const auto solve = [&]
    {
        cg::Vector x(31);
        cg::Workspace workspace(31);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
        return std::make_pair(info, std::move(x));
    };
    auto first = std::async(std::launch::async, solve);
    auto second = std::async(std::launch::async, solve);
    const auto first_result = first.get();
    const auto second_result = second.get();
    EXPECT_EQ(first_result.first.status, cg::Status::converged);
    EXPECT_EQ(second_result.first.status, cg::Status::converged);
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        EXPECT_NEAR(first_result.second.data()[i], 1.0, 1e-7);
        EXPECT_DOUBLE_EQ(first_result.second.data()[i], second_result.second.data()[i]);
    }
}

} // namespace
