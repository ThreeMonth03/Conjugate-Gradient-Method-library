#include "cg/cg.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using cg::ConstVectorView;
using cg::VectorView;

cg::DenseMatrix tridiagonal(std::size_t n) {
    cg::DenseMatrix A(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        A.data()[i * n + i] = 2.0;
        if (i != 0) {
            A.data()[i * n + i - 1] = -1.0;
        }
        if (i + 1 != n) {
            A.data()[i * n + i + 1] = -1.0;
        }
    }
    return A;
}

TEST(Storage, OwnsItsValuesAndHasConstViews) {
    std::vector<double> source{1.0, 2.0, 3.0};
    cg::Vector owned(source);
    source[0] = 100.0;
    EXPECT_EQ(owned.data()[0], 1.0);
    const auto& immutable = owned;
    ConstVectorView view = immutable.view();
    EXPECT_EQ(view.size, 3);
    EXPECT_EQ(view.data[2], 3.0);
}

TEST(Storage, ChecksShapeAndOverflowBeforeAccess) {
    EXPECT_THROW((cg::DenseMatrix(2, 3, {1.0})), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(nullptr, 2, 2)), std::invalid_argument);
    double value = 0.0;
    EXPECT_THROW((cg::DenseMatrixView(&value, 2, 3, 2)), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(&value, std::numeric_limits<std::size_t>::max(), 2)),
                 std::invalid_argument);
}

TEST(Workspace, ReusesStorageAcrossSameAndSmallerSizes) {
    cg::Workspace workspace(32);
    const auto* pointer = workspace.residual().data;
    const auto capacity = workspace.capacity();
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data, pointer);
    workspace.prepare(5);
    EXPECT_EQ(workspace.residual().data, pointer);
    EXPECT_EQ(workspace.capacity(), capacity);
    EXPECT_EQ(workspace.size(), 5);
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data, pointer);
    workspace.prepare(64);
    EXPECT_EQ(workspace.size(), 64);
    EXPECT_GE(workspace.capacity(), 64);
}

TEST(Backend, ValidatesThreadCountAndKeepsContextLocal) {
    EXPECT_THROW(cg::CpuBackend(0), std::invalid_argument);
    EXPECT_THROW(cg::CpuBackend(-1), std::invalid_argument);
    const cg::CpuBackend first(1, false);
    const cg::CpuBackend second(1, true);
    EXPECT_FALSE(first.fused());
    EXPECT_TRUE(second.fused());
    if (!cg::CpuBackend::openmp_enabled()) {
        EXPECT_THROW(cg::CpuBackend(2), std::invalid_argument);
    } else {
        const cg::CpuBackend parallel(2);
        EXPECT_EQ(first.threads(), 1);
        EXPECT_EQ(parallel.threads(), 2);
    }
}

TEST(Backend, HandlesPaddedRowMajorStorage) {
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

TEST(Backend, SupportsExactInPlaceAxpbyAndRejectsPartialOverlap) {
    cg::Vector x({1, 2, 3});
    cg::Vector y({4, 5, 6});
    const cg::CpuBackend backend;
    backend.axpby(2.0, x.view(), 3.0, y.view(), y.view());
    EXPECT_DOUBLE_EQ(y.data()[0], 14);
    EXPECT_DOUBLE_EQ(y.data()[2], 24);
    backend.axpy(2.0, x.view(), x.view());
    EXPECT_DOUBLE_EQ(x.data()[2], 9);
    EXPECT_THROW(backend.copy({x.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
    EXPECT_THROW(backend.axpy(1.0, {x.data(), 2}, {x.data() + 1, 2}),
                 std::invalid_argument);
    EXPECT_THROW(backend.axpby(1.0, {x.data(), 2}, 1.0, {y.data(), 2},
                              {x.data() + 1, 2}),
                 std::invalid_argument);
}

TEST(Backend, RejectsDimensionsNullPointersAndMatvecAliasing) {
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

TEST(Backend, ResidualSignIsExplicit) {
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

TEST(CGExplicit, SolvesFromAnIndependentInitialGuess) {
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({1, 2});
    cg::Vector x({0.5, -0.25});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.rtol = 1e-12;
    const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_GT(info.iterations, 0);
    EXPECT_EQ(info.matvecs, 1 + 2 * info.iterations);
    EXPECT_NEAR(x.data()[0], 1.0 / 11, 1e-12);
    EXPECT_NEAR(x.data()[1], 7.0 / 11, 1e-12);
    EXPECT_LE(info.residual_norm, options.rtol * std::sqrt(5.0));
}

TEST(CGExplicit, ChecksInitialResidualAndZeroRightHandSide) {
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({5, 4});
    cg::Vector x({1, 1});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.matvecs, 1);
}

TEST(CGExplicit, ZeroRightHandSideConvergesWithoutAnIteration) {
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b(2);
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    const auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.residual_norm, 0);
}

TEST(CGExplicit, ReportsIterationLimitAndTrueResidual) {
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    cg::Vector b({1, 2});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.maxiter = 0;
    auto info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.matvecs, 1);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(5.0));
    options.maxiter = 1;
    info = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 1);
    cg::Vector check(2);
    EXPECT_DOUBLE_EQ(info.residual_norm,
                     std::sqrt(backend.residual(A.view(), b.view(), x.view(), check.view())));
}

TEST(CGExplicit, ReportsBreakdownAndNonfiniteData) {
    cg::DenseMatrix indefinite(2, 2, {-1, 0, 0, -1});
    cg::Vector b({1, 2});
    cg::Vector x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    auto info = cg::solve_into(indefinite.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0);
    EXPECT_EQ(info.matvecs, 2);
    indefinite.data()[0] = std::numeric_limits<double>::quiet_NaN();
    info = cg::solve_into(indefinite.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(info.status, cg::Status::non_finite);
}

TEST(CGExplicit, ValidatesOptionsStorageAndWorkspace) {
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

TEST(CGExplicit, ReusingWorkspaceResetsAllIterationState) {
    const auto A = tridiagonal(31);
    cg::Vector truth(31);
    for (std::size_t i = 0; i < truth.size(); ++i) {
        truth.data()[i] = 0.25 + std::sin(static_cast<double>(i));
    }
    cg::Vector b(31);
    cg::Vector x(31);
    cg::Workspace workspace(31);
    const cg::CpuBackend backend;
    backend.apply(A.view(), truth.view(), b.view());
    const auto* pointer = workspace.residual().data;
    const auto first = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(first.status, cg::Status::converged);
    for (std::size_t i = 0; i < x.size(); ++i) {
        EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-6);
        x.data()[i] = 0;
    }
    const auto second = cg::solve_into(A.view(), b.view(), x.view(), workspace, backend);
    EXPECT_EQ(second.status, cg::Status::converged);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_EQ(workspace.residual().data, pointer);
}

} // namespace
