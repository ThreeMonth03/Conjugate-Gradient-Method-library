#include <cg/cg.hpp>
#include <cg/storage/detail/validation.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

using cg::ConstVectorView;
using cg::VectorView;

template <class View>
constexpr bool writable_dense_metadata = requires(View view) { view.data = nullptr; } || requires(View view) {
    view.rows = std::size_t{1};
} || requires(View view) { view.cols = std::size_t{1}; } || requires(View view) { view.row_stride = std::size_t{1}; };

static_assert(!writable_dense_metadata<cg::DenseMatrixView>);
static_assert(std::is_trivially_copyable_v<cg::DenseMatrixView>);
static_assert(std::is_standard_layout_v<cg::DenseMatrixView>);

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

TEST(Storage, ChecksShapeAndOverflowBeforeAccess)
{
    EXPECT_THROW((cg::DenseMatrix(2, 3, {1.0})), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(nullptr, 2, 2)), std::invalid_argument);
    double value = 0.0;
    EXPECT_THROW((cg::DenseMatrixView(&value, 2, 3, 2)), std::invalid_argument);
    EXPECT_THROW((cg::DenseMatrixView(&value, std::numeric_limits<std::size_t>::max(), 2)), std::invalid_argument);
}

TEST(Storage, DenseMatrixViewsCopyBorrowedStorageWithReadOnlyMetadata)
{
    const cg::DenseMatrixView empty;
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.rows(), 0u);
    EXPECT_EQ(empty.cols(), 0u);
    EXPECT_EQ(empty.row_stride(), 0u);

    std::array<double, 6> values{1, 2, 99, 3, 4, 99};
    const cg::DenseMatrixView padded(values.data(), 2, 2, 3);
    auto copied = padded;
    EXPECT_EQ(copied.data(), values.data());
    EXPECT_EQ(copied.rows(), 2u);
    EXPECT_EQ(copied.cols(), 2u);
    EXPECT_EQ(copied.row_stride(), 3u);
    values[0] = 7;
    EXPECT_DOUBLE_EQ(copied.data()[0], 7);

    copied = cg::DenseMatrixView(values.data(), 2, 3);
    EXPECT_EQ(copied.row_stride(), 3u);
    EXPECT_EQ(padded.cols(), 2u);
}

TEST(Storage, RejectsUnalignedViewsBeforeReadingTheirData)
{
    alignas(double) unsigned char storage[sizeof(double) * 4 + 1] = {};
    const auto *unaligned = reinterpret_cast<const double *>(storage + 1);
    EXPECT_THROW((cg::DenseMatrixView(unaligned, 2, 2)), std::invalid_argument);
    EXPECT_THROW(cg::detail::validate_raw_vector(unaligned, 1), std::invalid_argument);
}

TEST(Storage, RejectsRawSizeAndAddressOverflowBeforeConstructingASpan)
{
    double value = 1.0;
    EXPECT_THROW(cg::detail::validate_raw_vector(&value, std::numeric_limits<std::size_t>::max()),
                 std::invalid_argument);
    const auto address = std::numeric_limits<std::uintptr_t>::max() & ~(alignof(double) - 1);
    const auto *near_end = reinterpret_cast<const double *>(address);
    EXPECT_THROW(cg::detail::validate_raw_vector(near_end, 1), std::invalid_argument);
    EXPECT_NO_THROW(cg::detail::validate_raw_vector(nullptr, 0));
}

TEST(Storage, DenseMatrixCopiesOwnValuesAndMovesLeaveAnEmptyShape)
{
    cg::DenseMatrix source(2, 2, {4.0, 1.0, 1.0, 3.0});
    cg::DenseMatrix copied(source);
    copied.data()[0] = 9.0;
    EXPECT_DOUBLE_EQ(source.data()[0], 4.0);
    const auto *storage = source.data();

    cg::DenseMatrix moved(std::move(source));
    EXPECT_EQ(moved.data(), storage);
    EXPECT_EQ(moved.rows(), 2u);
    EXPECT_EQ(moved.cols(), 2u);
    EXPECT_DOUBLE_EQ(moved.data()[3], 3.0);
    EXPECT_EQ(source.rows(), 0u);
    EXPECT_EQ(source.cols(), 0u);
    EXPECT_NO_THROW(source.view());

    cg::DenseMatrix assigned(1, 1, {0.0});
    assigned = std::move(moved);
    EXPECT_EQ(assigned.data(), storage);
    EXPECT_EQ(assigned.rows(), 2u);
    EXPECT_EQ(assigned.cols(), 2u);
    EXPECT_EQ(moved.rows(), 0u);
    EXPECT_EQ(moved.cols(), 0u);
    EXPECT_NO_THROW(moved.view());
    auto &alias = assigned;
    assigned = std::move(alias);
    EXPECT_EQ(assigned.data(), storage);
    EXPECT_EQ(assigned.rows(), 2u);

    copied = assigned;
    copied.data()[3] = 7.0;
    EXPECT_DOUBLE_EQ(assigned.data()[3], 3.0);
}

static_assert(!std::is_copy_constructible_v<cg::Workspace>);
static_assert(!std::is_copy_assignable_v<cg::Workspace>);
static_assert(std::is_nothrow_move_constructible_v<cg::Workspace>);
static_assert(std::is_nothrow_move_assignable_v<cg::Workspace>);

TEST(Workspace, MovesPreserveAllBuffersAndLeaveReusableEmptySources)
{
    const auto buffers = [](cg::Workspace &workspace)
    {
        return std::array<cg::VectorView, 5>{workspace.residual(), workspace.direction(), workspace.product(),
                                             workspace.scratch(), workspace.preconditioned()};
    };
    cg::Workspace source(8);
    const auto original = buffers(source);
    for (std::size_t index = 0; index < original.size(); ++index)
    {
        original[index][0] = static_cast<double>(index + 1);
    }
    cg::Workspace moved(std::move(source));
    EXPECT_EQ(source.size(), 0u);
    for (const auto buffer : buffers(source))
    {
        EXPECT_TRUE(buffer.empty());
    }
    auto transferred = buffers(moved);
    for (std::size_t index = 0; index < transferred.size(); ++index)
    {
        EXPECT_EQ(transferred[index].data(), original[index].data());
        EXPECT_EQ(transferred[index].size(), 8u);
        EXPECT_DOUBLE_EQ(transferred[index][0], static_cast<double>(index + 1));
    }
    source.prepare(3);
    for (const auto buffer : buffers(source))
    {
        EXPECT_EQ(buffer.size(), 3u);
    }
    cg::Workspace assigned(2);
    assigned = std::move(moved);
    EXPECT_EQ(moved.size(), 0u);
    for (const auto buffer : buffers(moved))
    {
        EXPECT_TRUE(buffer.empty());
    }
    auto &alias = assigned;
    assigned = std::move(alias);
    transferred = buffers(assigned);
    for (std::size_t index = 0; index < transferred.size(); ++index)
    {
        EXPECT_EQ(transferred[index].data(), original[index].data());
        EXPECT_EQ(transferred[index].size(), 8u);
        EXPECT_DOUBLE_EQ(transferred[index][0], static_cast<double>(index + 1));
    }
    moved.prepare(4);
    for (const auto buffer : buffers(moved))
    {
        EXPECT_EQ(buffer.size(), 4u);
    }
}

TEST(Workspace, ReusesStorageAcrossSameAndSmallerSizes)
{
    cg::Workspace workspace(32);
    const auto *pointer = workspace.residual().data();
    const auto capacity = workspace.capacity();
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data(), pointer);
    workspace.prepare(5);
    EXPECT_EQ(workspace.residual().data(), pointer);
    EXPECT_EQ(workspace.capacity(), capacity);
    EXPECT_EQ(workspace.size(), 5u);
    workspace.prepare(32);
    EXPECT_EQ(workspace.residual().data(), pointer);
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
    const std::vector<double> input({2, 3});
    std::vector<double> output(2);
    const cg::CpuBackend backend;
    backend.apply(A, input, output);
    EXPECT_DOUBLE_EQ(output.data()[0], 8);
    EXPECT_DOUBLE_EQ(output.data()[1], 18);
    EXPECT_DOUBLE_EQ(backend.apply_and_dot(A, input, output), 70);
}

TEST(Backend, EmptyColumnsProduceZeroProductsWithoutDereferencingNullStorage)
{
    const cg::DenseMatrixView A(nullptr, 3, 0);
    const std::vector<double> b({1, 2, 3});
    std::vector<double> output({10, 20, 30});
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused);
        backend.apply(A, {}, output);
        for (std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(output.data()[i], 0);
        }
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A, b, {}, output), 14);
        EXPECT_DOUBLE_EQ(output.data()[1], 2);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A, b, {}, output, cg::ResidualSign::product_minus_rhs), 14);
        EXPECT_DOUBLE_EQ(output.data()[1], -2);
    }
}

TEST(Backend, SupportsExactInPlaceAxpbyAndRejectsPartialOverlap)
{
    std::vector<double> x({1, 2, 3});
    std::vector<double> y({4, 5, 6});
    const cg::CpuBackend backend;
    backend.axpby(2.0, x, 3.0, y, y);
    EXPECT_DOUBLE_EQ(y.data()[0], 14);
    EXPECT_DOUBLE_EQ(y.data()[2], 24);
    backend.axpy(2.0, x, x);
    EXPECT_DOUBLE_EQ(x.data()[2], 9);
    EXPECT_THROW(backend.copy({x.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
    EXPECT_THROW(backend.axpy(1.0, {x.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
    EXPECT_THROW(backend.axpby(1.0, {x.data(), 2}, 1.0, {y.data(), 2}, {x.data() + 1, 2}), std::invalid_argument);
}

TEST(Backend, RejectsDimensionsNullPointersAndMatvecAliasing)
{
    cg::DenseMatrix A(2, 2, {1, 0, 0, 1});
    std::vector<double> x({1, 2});
    std::vector<double> y(2);
    const cg::CpuBackend backend;
    EXPECT_THROW(backend.dot(x, {y.data(), 1}), std::invalid_argument);
    EXPECT_THROW(cg::detail::validate_raw_vector(nullptr, 2), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x, x), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x, {A.data(), 2}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(backend.dot({}, {}), 0.0);
}

TEST(Backend, ResidualSignIsExplicit)
{
    cg::DenseMatrix A(2, 2, {2, 0, 0, 3});
    std::vector<double> b({5, 9});
    std::vector<double> x({1, 2});
    std::vector<double> r(2);
    const cg::CpuBackend backend;
    EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b, x, r), 18);
    EXPECT_DOUBLE_EQ(r.data()[0], 3);
    EXPECT_DOUBLE_EQ(r.data()[1], 3);
    EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b, x, r, cg::ResidualSign::product_minus_rhs), 18);
    EXPECT_DOUBLE_EQ(r.data()[0], -3);
}

TEST(Backend, StableNormHandlesTinyLargeAndSubnormalValues)
{
    const cg::CpuBackend backend;
    for (const double magnitude :
         {1e-200, 1e-160, 1e200, std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max() / 4.0})
    {
        const std::vector<double> values({magnitude, -magnitude});
        const double expected = std::hypot(magnitude, magnitude);
        const double ulp = std::nextafter(expected, std::numeric_limits<double>::infinity()) - expected;
        EXPECT_NEAR(backend.norm(values), expected, 8.0 * ulp);
    }
    const std::vector<double> zero(3);
    EXPECT_DOUBLE_EQ(backend.norm(zero), 0);
    EXPECT_DOUBLE_EQ(backend.norm({}), 0);
    const std::vector<double> infinite({std::numeric_limits<double>::infinity(), 1.0});
    const std::vector<double> nan({std::numeric_limits<double>::quiet_NaN(), 1.0});
    EXPECT_TRUE(std::isinf(backend.norm(infinite)));
    EXPECT_TRUE(std::isnan(backend.norm(nan)));
}

TEST(CGRange, DoesNotClaimConvergenceWhenSquaredResidualUnderflowsOrOverflows)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::CpuBackend backend;
    for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
    {
        for (const double magnitude : {1e-200, 1e-160, 1e200})
        {
            const std::vector<double> b({magnitude, -magnitude});
            std::vector<double> x(2);
            cg::Workspace workspace(2);
            cg::CgOptions options;
            options.algorithm = algorithm;
            const auto info = cg::solve_into(identity.view(), b, x, workspace, backend, options);
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
            const std::vector<double> b({magnitude, -magnitude});
            std::vector<double> x(2);
            cg::Workspace workspace(2);
            const auto info = cg::solve_into(identity.view(), b, x, workspace, backend, options);
            EXPECT_EQ(info.status, cg::Status::converged);
            EXPECT_EQ(info.iterations, 1u);
            EXPECT_DOUBLE_EQ(info.residual_norm, 0);
        }
        const std::vector<double> tiny({1e-200, -1e-200});
        std::vector<double> x(2);
        cg::Workspace workspace(2);
        options.atol = 1e-199;
        const auto accepted = cg::solve_into(identity.view(), tiny, x, workspace, backend, options);
        EXPECT_EQ(accepted.status, cg::Status::converged);
        EXPECT_EQ(accepted.iterations, 0u);
        EXPECT_GT(accepted.residual_norm, 0);
        EXPECT_LE(accepted.residual_norm, options.atol);
        options.atol = 0;
        options.maxiter = 0;
        const auto limited = cg::solve_into(identity.view(), tiny, x, workspace, backend, options);
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
            const std::vector<double> b(std::vector<double>{magnitude});
            std::vector<double> x(1);
            cg::Workspace workspace(1);
            cg::CgOptions options;
            options.algorithm = algorithm;
            const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
            EXPECT_EQ(info.status, cg::Status::non_finite);
            EXPECT_EQ(info.iterations, 0u);
            EXPECT_DOUBLE_EQ(info.residual_norm, magnitude);
        }
    }
}

TEST(CGExplicit, SolvesFromAnIndependentInitialGuess)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    std::vector<double> b({1, 2});
    std::vector<double> x({0.5, -0.25});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    options.rtol = 1e-12;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
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
    std::vector<double> b({5, 4});
    std::vector<double> x({1, 1});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
}

TEST(CGExplicit, ZeroRightHandSideConvergesWithoutAnIteration)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    std::vector<double> b(2);
    std::vector<double> x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.residual_norm, 0);
}

TEST(CGExplicit, ReportsIterationLimitAndTrueResidual)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    std::vector<double> b({1, 2});
    std::vector<double> x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    options.maxiter = 0;
    auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(5.0));
    options.maxiter = 1;
    info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 1u);
    std::vector<double> check(2);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(backend.residual_squared_norm(A.view(), b, x, check)));
}

TEST(CGExplicit, ReportsBreakdownAndNonfiniteData)
{
    cg::DenseMatrix indefinite(2, 2, {-1, 0, 0, -1});
    std::vector<double> b({1, 2});
    std::vector<double> x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    auto info = cg::solve_into(indefinite.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 2u);
    indefinite.data()[0] = std::numeric_limits<double>::quiet_NaN();
    info = cg::solve_into(indefinite.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::non_finite);
}

TEST(CGExplicit, ValidatesOptionsStorageAndWorkspace)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    std::vector<double> b({1, 2});
    std::vector<double> x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    options.rtol = -1;
    EXPECT_THROW(cg::solve_into(A.view(), b, x, workspace, backend, options), std::invalid_argument);
    options.rtol = std::numeric_limits<double>::infinity();
    EXPECT_THROW(cg::solve_into(A.view(), b, x, workspace, backend, options), std::invalid_argument);
    options.rtol = 1e-8;
    EXPECT_THROW(cg::solve_into(A.view(), b, b, workspace, backend, options), std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A.view(), b, {A.data(), 2}, workspace, backend, options), std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A.view(), b, workspace.residual(), workspace, backend, options), std::invalid_argument);
    workspace.prepare(1);
    EXPECT_THROW(cg::solve_into(A.view(), b, x, workspace, backend, options), std::invalid_argument);
}

TEST(CGExplicit, ReusingWorkspaceResetsAllIterationState)
{
    const auto A = tridiagonal(31);
    std::vector<double> truth(31);
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        truth.data()[i] = 0.25 + std::sin(static_cast<double>(i));
    }
    std::vector<double> b(31);
    std::vector<double> x(31);
    cg::Workspace workspace(31);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    backend.apply(A.view(), truth, b);
    const auto *pointer = workspace.residual().data();
    const auto first = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(first.status, cg::Status::converged);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-6);
        x.data()[i] = 0;
    }
    const auto second = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(second.status, cg::Status::converged);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_EQ(workspace.residual().data(), pointer);
}

TEST(FusedOperations, MatchesReferenceVectorUpdatesIncludingSimdTails)
{
    for (const std::size_t size : {1u, 3u, 17u, 127u, 16385u})
    {
        std::vector<double> x(size), r(size), p(size), ap(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            x.data()[i] = std::sin(static_cast<double>(i));
            r.data()[i] = std::cos(static_cast<double>(i));
            p.data()[i] = 0.1 + static_cast<double>(i % 7);
            ap.data()[i] = -0.2 + static_cast<double>(i % 11);
        }
        std::vector<double> reference_x = x;
        std::vector<double> reference_r = r;
        const cg::CpuBackend fused(1, true);
        const cg::CpuBackend reference(1, false);
        const auto actual = fused.update_x_r_squared_norm(x, r, p, ap, 0.25);
        const auto expected = reference.update_x_r_squared_norm(reference_x, reference_r, p, ap, 0.25);
        EXPECT_NEAR(actual, expected, std::max(1.0, expected) * 1e-12);
        for (std::size_t i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(x.data()[i], reference_x.data()[i]);
            EXPECT_DOUBLE_EQ(r.data()[i], reference_r.data()[i]);
        }
        EXPECT_THROW(fused.update_x_r_squared_norm(x, x, p, ap, 0.25), std::invalid_argument);
    }
}

TEST(FusedOperations, MatchesReferenceDenseReductionAndResidual)
{
    for (const std::size_t size : {1u, 3u, 17u, 129u})
    {
        const auto A = tridiagonal(size);
        std::vector<double> p(size), b(size), ap(size), reference_ap(size), r(size), reference_r(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            p.data()[i] = std::sin(static_cast<double>(i)) + 0.5;
            b.data()[i] = std::cos(static_cast<double>(i));
        }
        const int threads = cg::CpuBackend::openmp_enabled() ? 2 : 1;
        const cg::CpuBackend fused(threads, true);
        const cg::CpuBackend reference(threads, false);
        EXPECT_NEAR(fused.apply_and_dot(A.view(), p, ap), reference.apply_and_dot(A.view(), p, reference_ap), 1e-10);
        EXPECT_NEAR(fused.residual_squared_norm(A.view(), b, p, r),
                    reference.residual_squared_norm(A.view(), b, p, reference_r), 1e-10);
        for (std::size_t i = 0; i < size; ++i)
        {
            const double tolerance = 16.0 * std::numeric_limits<double>::epsilon();
            EXPECT_NEAR(ap.data()[i], reference_ap.data()[i], tolerance);
            EXPECT_NEAR(r.data()[i], reference_r.data()[i], tolerance);
        }
    }
}

TEST(DenseRows, NativeMatvecIsIndependentOfFusionSelection)
{
    // Cancellation makes different GEMV accumulation policies observable.
    // Include a row pair, a leftover row, a column tail and poisoned padding.
    constexpr std::size_t rows = 3, columns = 5, stride = 8;
    const std::array<double, rows * columns> values{1e16, 1, -1e16, 1, 4, -1e16, 1, 1e16, 1, -3, 1e16, 1, -1e16, 1, 7};
    std::array<double, rows * stride> storage;
    storage.fill(std::numeric_limits<double>::quiet_NaN());
    for (std::size_t row = 0; row < rows; ++row)
    {
        for (std::size_t column = 0; column < columns; ++column)
        {
            storage[row * stride + column] = values[row * columns + column];
        }
    }
    const cg::DenseMatrixView A(storage.data(), rows, columns, stride);
    const std::array<double, columns> x{1, 1, 1, 1, 1};
    std::array<double, rows> fused_output{}, separate_output{};
    cg::CpuBackend(1, true).apply(A, x, fused_output);
    cg::CpuBackend(1, false).apply(A, x, separate_output);
    for (std::size_t row = 0; row < rows; ++row)
    {
        EXPECT_TRUE(std::isfinite(fused_output[row]));
        EXPECT_EQ(separate_output[row], fused_output[row]);
    }
}

TEST(DenseRows, FusionPoliciesMatchLongDoubleWithOddTailsAndPaddedRows)
{
    for (const std::size_t columns : {1u, 3u, 5u, 7u, 17u, 65u, 129u, 257u})
    {
        const std::size_t rows = 3;
        const std::size_t stride = columns + 5;
        std::vector<double> storage(rows * stride, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> x(columns), output(rows);
        for (std::size_t column = 0; column < columns; ++column)
        {
            x.data()[column] = std::sin(static_cast<double>(column)) + 0.5;
            for (std::size_t row = 0; row < rows; ++row)
            {
                storage[row * stride + column] =
                    std::cos(static_cast<double>(row + column)) / static_cast<double>(columns);
            }
        }
        const cg::DenseMatrixView A(storage.data(), rows, columns, stride);
        for (const bool fused : {false, true})
        {
            const cg::CpuBackend backend(1, fused);
            backend.apply(A, x, output);
            for (std::size_t row = 0; row < rows; ++row)
            {
                long double expected = 0;
                long double absolute_sum = 0;
                for (std::size_t column = 0; column < columns; ++column)
                {
                    const long double product = static_cast<long double>(storage[row * stride + column]) *
                                                static_cast<long double>(x.data()[column]);
                    expected += product;
                    absolute_sum += std::abs(product);
                }
                const double rounding_bound = static_cast<double>(absolute_sum) *
                                              std::numeric_limits<double>::epsilon() * static_cast<double>(columns + 8);
                EXPECT_TRUE(std::isfinite(output.data()[row]));
                EXPECT_NEAR(output.data()[row], static_cast<double>(expected), rounding_bound);
            }
        }
    }
}

TEST(DenseRows, FusionPoliciesPreserveNonfiniteMultiplicationBehavior)
{
    cg::DenseMatrix A(17, 17);
    std::vector<double> x(17), b(17), output(17);
    for (std::size_t i = 0; i < 17; ++i)
    {
        A.data()[i * 17 + i] = 1;
    }
    A.data()[16] = std::numeric_limits<double>::infinity();
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused);
        backend.apply(A.view(), x, output);
        EXPECT_TRUE(std::isnan(output.data()[0]));
        EXPECT_TRUE(std::isnan(backend.apply_and_dot(A.view(), x, output)));
        EXPECT_TRUE(std::isnan(backend.residual_squared_norm(A.view(), b, x, output)));
    }
}

TEST(DenseRows, PairBoundariesPreserveExactProductsAndBothResidualSigns)
{
    // Fully populated integer products stay exactly representable in double.
    // The odd size exercises a leftover row after parallel pairs; the even
    // size exercises the all-pair path. NaN padding detects stride overreads.
    for (const std::size_t size : {129u, 130u})
    {
        SCOPED_TRACE(size);
        const auto stride = size + 5;
        std::vector<double> storage(size * stride, std::numeric_limits<double>::quiet_NaN());
        std::vector<double> p(size), b(size), output(size), expected_product(size);
        for (std::size_t i = 0; i < size; ++i)
        {
            p[i] = static_cast<double>(static_cast<int>(i % 5) - 2);
            b[i] = static_cast<double>(static_cast<int>(i % 7) - 3);
            for (std::size_t j = 0; j < size; ++j)
            {
                storage[i * stride + j] = static_cast<double>(static_cast<int>((i + 3 * j) % 7) - 3);
            }
        }
        long double expected_dot = 0;
        for (std::size_t i = 0; i < size; ++i)
        {
            long double sum = 0;
            for (std::size_t j = 0; j < size; ++j)
            {
                sum += static_cast<long double>(storage[i * stride + j]) * static_cast<long double>(p[j]);
            }
            expected_product[i] = static_cast<double>(sum);
            expected_dot += static_cast<long double>(p[i]) * sum;
        }
        const cg::DenseMatrixView A(storage.data(), size, size, stride);
        for (const int threads : {1, 2})
        {
            if (threads > 1 && !cg::CpuBackend::openmp_enabled())
            {
                continue;
            }
            SCOPED_TRACE(threads);
            for (const bool fused : {false, true})
            {
                SCOPED_TRACE(fused);
                const cg::CpuBackend backend(threads, fused);
                backend.apply(A, p, output);
                for (std::size_t i = 0; i < size; ++i)
                {
                    EXPECT_DOUBLE_EQ(output[i], expected_product[i]);
                }
                EXPECT_DOUBLE_EQ(backend.apply_and_dot(A, p, output), static_cast<double>(expected_dot));
                for (std::size_t i = 0; i < size; ++i)
                {
                    EXPECT_DOUBLE_EQ(output[i], expected_product[i]);
                }
                for (const auto sign : {cg::ResidualSign::rhs_minus_product, cg::ResidualSign::product_minus_rhs})
                {
                    const double actual_norm = backend.residual_squared_norm(A, b, p, output, sign);
                    long double expected_norm = 0;
                    for (std::size_t i = 0; i < size; ++i)
                    {
                        const long double expected = sign == cg::ResidualSign::rhs_minus_product
                                                         ? static_cast<long double>(b[i]) - expected_product[i]
                                                         : expected_product[i] - static_cast<long double>(b[i]);
                        EXPECT_DOUBLE_EQ(output[i], static_cast<double>(expected));
                        expected_norm += expected * expected;
                    }
                    EXPECT_DOUBLE_EQ(actual_norm, static_cast<double>(expected_norm));
                }
            }
        }
    }
}

TEST(DenseRows, NonfiniteValuesStayInTheAffectedPairOrLeftoverRow)
{
    constexpr std::size_t size = 129;
    constexpr std::size_t stride = size + 5;
    for (const std::size_t affected_row : {1u, 128u})
    {
        SCOPED_TRACE(affected_row);
        for (const std::size_t affected_column : {3u, 128u})
        {
            SCOPED_TRACE(affected_column);
            for (const double bad_value :
                 {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
                  -std::numeric_limits<double>::infinity()})
            {
                std::vector<double> storage(size * stride, std::numeric_limits<double>::quiet_NaN());
                std::vector<double> zero(size), output(size);
                for (std::size_t i = 0; i < size; ++i)
                {
                    for (std::size_t j = 0; j < size; ++j)
                    {
                        storage[i * stride + j] = i == j ? 1.0 : 0.0;
                    }
                }
                storage[affected_row * stride + affected_column] = bad_value;
                const cg::DenseMatrixView A(storage.data(), size, size, stride);
                for (const int threads : {1, 2})
                {
                    if (threads > 1 && !cg::CpuBackend::openmp_enabled())
                    {
                        continue;
                    }
                    for (const bool fused : {false, true})
                    {
                        const cg::CpuBackend backend(threads, fused);
                        backend.apply(A, zero, output);
                        for (std::size_t i = 0; i < size; ++i)
                        {
                            if (i == affected_row)
                            {
                                EXPECT_TRUE(std::isnan(output[i]));
                            }
                            else
                            {
                                EXPECT_DOUBLE_EQ(output[i], 0.0);
                            }
                        }
                        EXPECT_TRUE(std::isnan(backend.apply_and_dot(A, zero, output)));
                        EXPECT_TRUE(std::isnan(backend.residual_squared_norm(A, zero, zero, output)));
                        EXPECT_TRUE(std::isnan(output[affected_row]));
                    }
                }
            }
        }
    }
}

TEST(CGRecurrence, SolvesAndVerifiesTrueResidualWithBothKernelPolicies)
{
    const auto A = tridiagonal(31);
    std::vector<double> truth(31), b(31);
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        truth.data()[i] = 0.25 + std::sin(static_cast<double>(i));
    }
    const cg::CpuBackend reference;
    reference.apply(A.view(), truth, b);
    for (const bool fused : {false, true})
    {
        std::vector<double> x(31), check(31);
        cg::Workspace workspace(31);
        const cg::CpuBackend backend(1, fused);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        options.residual_refresh = 0;
        options.rtol = 1e-10;
        const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
        EXPECT_EQ(info.status, cg::Status::converged);
        EXPECT_EQ(info.restarts, 0u);
        EXPECT_EQ(info.matvecs, 2 + info.iterations);
        EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(backend.residual_squared_norm(A.view(), b, x, check)));
        EXPECT_LE(info.residual_norm, options.rtol * std::sqrt(backend.squared_norm(b)));
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-7);
        }
    }
}

TEST(CGRecurrence, CountsRefreshMatvecsAndRestartsDirections)
{
    cg::DenseMatrix A(3, 3, {2, 0, 0, 0, 3, 0, 0, 0, 4});
    std::vector<double> b({1, 2, 3});
    std::vector<double> x(3), check(3);
    cg::Workspace workspace(3);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.rtol = 0;
    options.atol = 0;
    options.residual_refresh = 1;
    options.maxiter = 3;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.iterations, 3u);
    EXPECT_EQ(info.restarts, 2u);
    EXPECT_EQ(info.matvecs, 7u);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(backend.residual_squared_norm(A.view(), b, x, check)));
}

TEST(CGRecurrence, ChecksInitialGuessIterationLimitAndBreakdown)
{
    cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    std::vector<double> b({5, 4}), x({1, 1});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
    x.data()[0] = x.data()[1] = 0;
    options.maxiter = 0;
    info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.matvecs, 1u);
    options.maxiter = 1;
    info = cg::solve_into(A.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.matvecs, 3u);
    cg::DenseMatrix indefinite(2, 2, {-1, 0, 0, -1});
    x.data()[0] = x.data()[1] = 0;
    info = cg::solve_into(indefinite.view(), b, x, workspace, backend, options);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 3u);
}

TEST(CGRecurrence, DistinctWorkspacesCanSolveOnOneContextConcurrently)
{
    const auto A = tridiagonal(31);
    std::vector<double> b(31);
    b.data()[0] = b.data()[30] = 1.0;
    const cg::CpuBackend backend;
    const auto solve = [&]
    {
        std::vector<double> x(31);
        cg::Workspace workspace(31);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
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
