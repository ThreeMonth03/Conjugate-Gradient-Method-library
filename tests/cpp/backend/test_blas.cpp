#include <cg/backend/cpu/detail/blas.hpp>
#include <cg/cg.hpp>
#include <cg/storage/detail/validation.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

class Blas : public testing::Test
{
    protected:
    void SetUp() override
    {
        if (!cg::CpuBackend::blas_enabled())
        {
            GTEST_SKIP() << "BLAS is not enabled in this build";
        }
    }
};

cg::DenseMatrix diagonal_spd(std::size_t size)
{
    cg::DenseMatrix A(size, size);
    for (std::size_t i = 0; i < size; ++i)
    {
        A.data()[i * size + i] = 1.0 + static_cast<double>(i);
        if (i + 1 != size)
        {
            A.data()[i * size + i + 1] = -0.125;
            A.data()[(i + 1) * size + i] = -0.125;
        }
    }
    return A;
}

double independent_residual(cg::DenseMatrixView A, cg::ConstVectorView b, cg::ConstVectorView x)
{
    double norm = 0.0;
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        long double product = 0.0L;
        for (std::size_t column = 0; column < A.cols(); ++column)
        {
            product += static_cast<long double>(A.data()[row * A.row_stride() + column]) *
                       static_cast<long double>(x.data()[column]);
        }
        norm = std::hypot(norm, static_cast<double>(static_cast<long double>(b.data()[row]) - product));
    }
    return norm;
}

TEST(BlasConfiguration, NativeRemainsDefaultAndBlasRequiresAnEnabledSingleThreadBuild)
{
    const cg::CpuBackend native;
    EXPECT_EQ(native.dense_kernel(), cg::DenseKernel::native);
    EXPECT_EQ(native.threads(), 1);
    EXPECT_TRUE(native.fused());
    EXPECT_THROW(cg::CpuBackend(1, true, static_cast<cg::DenseKernel>(-1)), std::invalid_argument);
    EXPECT_THROW(cg::CpuBackend(0, true, cg::DenseKernel::blas), std::invalid_argument);
    EXPECT_THROW(cg::CpuBackend(2, true, cg::DenseKernel::blas), std::invalid_argument);
    ASSERT_NE(cg::CpuBackend::blas_info(), nullptr);
    if (cg::CpuBackend::blas_enabled())
    {
        const cg::CpuBackend blas(1, false, cg::DenseKernel::blas);
        EXPECT_EQ(blas.dense_kernel(), cg::DenseKernel::blas);
        EXPECT_FALSE(blas.fused());
        EXPECT_STRNE(cg::CpuBackend::blas_info(), "disabled");
    }
    else
    {
        EXPECT_STREQ(cg::CpuBackend::blas_info(), "disabled");
        EXPECT_THROW(cg::CpuBackend(1, true, cg::DenseKernel::blas), std::invalid_argument);
    }
}

TEST_F(Blas, HandlesRectangularOddDimensionsAndPaddedRowsWithoutReadingOutput)
{
    const std::size_t rows = 3, columns = 17, stride = columns + 3;
    std::vector<double> storage(rows * stride, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> x(columns), output(rows);
    for (std::size_t column = 0; column < columns; ++column)
    {
        x.data()[column] = static_cast<double>(column % 7) - 3.0;
        for (std::size_t row = 0; row < rows; ++row)
        {
            storage[row * stride + column] = static_cast<double>((row + column) % 5) - 2.0;
        }
    }
    const cg::DenseMatrixView A(storage.data(), rows, columns, stride);
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
        std::fill_n(output.data(), rows, std::numeric_limits<double>::quiet_NaN());
        backend.apply(A, x, output);
        for (std::size_t row = 0; row < rows; ++row)
        {
            long double expected = 0.0L;
            for (std::size_t column = 0; column < columns; ++column)
            {
                expected += static_cast<long double>(storage[row * stride + column]) *
                            static_cast<long double>(x.data()[column]);
            }
            EXPECT_DOUBLE_EQ(output.data()[row], static_cast<double>(expected));
        }
    }
}

TEST_F(Blas, ReducesTheActualProductAndPreservesBothResidualSigns)
{
    const cg::DenseMatrix A(3, 3, {4, 1, 0, 1, 5, -1, 0, -1, 3});
    const std::vector<double> p({2, -1, 3}), b({8, -2, 11});
    std::vector<double> output(3);
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
        EXPECT_DOUBLE_EQ(backend.apply_and_dot(A.view(), p, output), 50);
        EXPECT_DOUBLE_EQ(output.data()[0], 7);
        EXPECT_DOUBLE_EQ(output.data()[1], -6);
        EXPECT_DOUBLE_EQ(output.data()[2], 10);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b, p, output), 18);
        EXPECT_DOUBLE_EQ(output.data()[0], 1);
        EXPECT_DOUBLE_EQ(output.data()[1], 4);
        EXPECT_DOUBLE_EQ(output.data()[2], 1);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b, p, output, cg::ResidualSign::product_minus_rhs),
                         18);
        EXPECT_DOUBLE_EQ(output.data()[0], -1);
        EXPECT_DOUBLE_EQ(output.data()[1], -4);
        EXPECT_DOUBLE_EQ(output.data()[2], -1);
    }
}

TEST_F(Blas, EmptyMatricesDoNotPassNullStorageToTheProvider)
{
    const cg::DenseMatrixView no_columns(nullptr, 3, 0), no_rows(nullptr, 0, 4), empty(nullptr, 0, 0);
    const std::vector<double> b({1, -2, 3}), x(4);
    std::vector<double> output({9, 9, 9});
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
        backend.apply(no_columns, {}, output);
        for (std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_DOUBLE_EQ(output.data()[i], 0);
        }
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(no_columns, b, {}, output), 14);
        EXPECT_DOUBLE_EQ(output.data()[1], -2);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(no_columns, b, {}, output, cg::ResidualSign::product_minus_rhs),
                         14);
        EXPECT_DOUBLE_EQ(output.data()[1], 2);
        EXPECT_NO_THROW(backend.apply(no_rows, x, {}));
        EXPECT_DOUBLE_EQ(backend.apply_and_dot(empty, {}, {}), 0);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(empty, {}, {}, {}), 0);
    }
}

TEST_F(Blas, RejectsInvalidShapesAndAliasingBeforeChangingOutput)
{
    cg::DenseMatrix A(2, 2, {2, 0, 0, 3});
    std::vector<double> x({1, 2}), output({7, 8});
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
    EXPECT_THROW(backend.apply(A.view(), {x.data(), 1}, output), std::invalid_argument);
    EXPECT_THROW(cg::detail::validate_raw_vector(nullptr, 2), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x, x), std::invalid_argument);
    EXPECT_THROW(backend.apply(A.view(), x, {A.data(), 2}), std::invalid_argument);
    EXPECT_THROW(backend.apply_and_dot(A.view(), x, x), std::invalid_argument);
    EXPECT_THROW(backend.residual_squared_norm(A.view(), output, x, output), std::invalid_argument);
    EXPECT_DOUBLE_EQ(output.data()[0], 7);
    EXPECT_DOUBLE_EQ(output.data()[1], 8);
}

TEST_F(Blas, RejectsInt32DimensionAndStrideOverflowBeforeAnyProviderAccess)
{
    const auto too_large = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    if (too_large == 0)
    {
        GTEST_SKIP() << "size_t cannot express an out-of-range BLAS dimension";
    }
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
    double a = 2, x = 3, output = 7;
    const cg::DenseMatrixView wide_stride(&a, 1, 1, too_large);
    EXPECT_THROW(backend.apply(wide_stride, {&x, 1}, {&output, 1}), std::invalid_argument);
    EXPECT_THROW(backend.apply_and_dot(wide_stride, {&x, 1}, {&output, 1}), std::invalid_argument);
    EXPECT_THROW(backend.residual_squared_norm(wide_stride, {&a, 1}, {&x, 1}, {&output, 1}), std::invalid_argument);
    // These intentionally oversized views must be rejected without reading
    // their data, including when the other matrix dimension is zero.
    const cg::DenseMatrixView wide(nullptr, 0, too_large), tall(nullptr, too_large, 0);
    EXPECT_THROW(cg::detail::blas_apply(wide, {}, {}), std::invalid_argument);
    EXPECT_THROW(cg::detail::blas_apply(tall, {}, {}), std::invalid_argument);
    EXPECT_DOUBLE_EQ(output, 7);
}

TEST_F(Blas, KeepsVectorFusionIndependentOfTheDenseKernel)
{
    const std::vector<double> p({2, -1, 3}), ap({7, -6, 10});
    for (const bool fused : {false, true})
    {
        std::vector<double> x({0, 0, 0}), r({8, -2, 11});
        const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
        EXPECT_DOUBLE_EQ(backend.update_x_r_squared_norm(x, r, p, ap, 1), 18);
        EXPECT_DOUBLE_EQ(x.data()[0], 2);
        EXPECT_DOUBLE_EQ(r.data()[1], 4);
        backend.update_direction(r, x, 2);
        EXPECT_DOUBLE_EQ(x.data()[0], 5);
        EXPECT_DOUBLE_EQ(x.data()[1], 2);
        EXPECT_DOUBLE_EQ(x.data()[2], 7);
    }
}

TEST_F(Blas, PrivateProviderPreservesNonfiniteProductsEvenForZeroInputs)
{
    const std::size_t size = 33;
    cg::DenseMatrix A(size, size);
    std::vector<double> zero(size), b(std::vector<double>(size, 1.0)), output(size);
    for (const double invalid : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()})
    {
        A.data()[16] = invalid;
        for (const bool fused : {false, true})
        {
            const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
            backend.apply(A.view(), zero, output);
            EXPECT_TRUE(std::isnan(output.data()[0]));
            EXPECT_TRUE(std::isnan(backend.apply_and_dot(A.view(), zero, output)));
            EXPECT_TRUE(std::isnan(backend.residual_squared_norm(A.view(), b, zero, output)));
            cg::Workspace workspace(size);
            for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
            {
                cg::CgOptions options;
                options.algorithm = algorithm;
                const auto info = cg::solve_into(A.view(), b, zero, workspace, backend, options);
                EXPECT_EQ(info.status, cg::Status::non_finite);
                EXPECT_EQ(info.iterations, 0u);
            }
        }
    }
}

TEST_F(Blas, DenseCgAndJacobiPcgPassIndependentTrueResidualChecksWithWorkspaceReuse)
{
    const std::size_t size = 17;
    const auto A = diagonal_spd(size);
    const cg::JacobiPreconditioner M(A.view());
    const std::vector<double> b(std::vector<double>(size, 1.0));
    std::vector<double> x(size);
    cg::Workspace workspace(size);
    const auto *allocation = workspace.residual().data();
    for (const bool fused : {false, true})
    {
        const cg::CpuBackend backend(1, fused, cg::DenseKernel::blas);
        for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
        {
            for (const bool preconditioned : {false, true})
            {
                if (preconditioned && algorithm != cg::Algorithm::recurrence)
                {
                    continue;
                }
                std::fill_n(x.data(), size, 0.0);
                cg::CgOptions options;
                options.algorithm = algorithm;
                options.rtol = 1e-10;
                options.maxiter = 200;
                options.residual_refresh = 7;
                const auto info =
                    cg::solve_into(A.view(), b, x, workspace, backend, options, preconditioned ? &M : nullptr);
                ASSERT_EQ(info.status, cg::Status::converged);
                EXPECT_LE(independent_residual(A.view(), b, x), options.rtol * std::sqrt(static_cast<double>(size)));
                EXPECT_EQ(workspace.residual().data(), allocation);
                const auto repeated =
                    cg::solve_into(A.view(), b, x, workspace, backend, options, preconditioned ? &M : nullptr);
                EXPECT_EQ(repeated.status, cg::Status::converged);
                EXPECT_EQ(repeated.iterations, 0u);
                EXPECT_EQ(repeated.matvecs, 1u);
            }
        }
    }
}

TEST_F(Blas, PreservesStableNormAndNumericalRangeStatuses)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::JacobiPreconditioner M(identity.view());
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
    for (const double magnitude : {1e-200, 1e-160, 1e200})
    {
        const std::vector<double> b({magnitude, -magnitude});
        std::vector<double> x(2);
        cg::Workspace workspace(2);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        for (const auto *preconditioner : {static_cast<const cg::JacobiPreconditioner *>(nullptr), &M})
        {
            const auto info = cg::solve_into(identity.view(), b, x, workspace, backend, options, preconditioner);
            EXPECT_EQ(info.status, cg::Status::non_finite);
            EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
        }
        options.maxiter = 0;
        const auto limited = cg::solve_into(identity.view(), b, x, workspace, backend, options);
        EXPECT_EQ(limited.status, cg::Status::max_iterations);
        EXPECT_NEAR(limited.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
    }
}

TEST_F(Blas, AcceptsActualConvergenceAndUsableResidualScaling)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
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
            EXPECT_DOUBLE_EQ(info.residual_norm, 0);
        }
        const std::vector<double> tiny({1e-200, -1e-200}), zero(2);
        std::vector<double> x(2);
        cg::Workspace workspace(2);
        options.atol = 1e-199;
        const auto accepted = cg::solve_into(identity.view(), tiny, x, workspace, backend, options);
        EXPECT_EQ(accepted.status, cg::Status::converged);
        EXPECT_GT(accepted.residual_norm, 0);
        EXPECT_EQ(accepted.iterations, 0u);
        const auto exact = cg::solve_into(identity.view(), zero, x, workspace, backend, options);
        EXPECT_EQ(exact.status, cg::Status::converged);
        EXPECT_DOUBLE_EQ(exact.residual_norm, 0);
    }
}

TEST_F(Blas, SharesAnImmutableBackendAcrossIndependentConcurrentSolves)
{
    const auto A = diagonal_spd(33);
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
    auto run = [&](double rhs)
    {
        const std::vector<double> b(std::vector<double>(A.rows(), rhs));
        std::vector<double> x(A.rows());
        cg::Workspace workspace(A.rows());
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        options.rtol = 1e-10;
        options.maxiter = 200;
        const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options);
        return std::make_pair(info.status, independent_residual(A.view(), b, x));
    };
    auto first = std::async(std::launch::async, run, 1.0);
    auto second = std::async(std::launch::async, run, -2.0);
    for (auto *future : {&first, &second})
    {
        const auto result = future->get();
        EXPECT_EQ(result.first, cg::Status::converged);
        EXPECT_LE(result.second, 2e-10 * std::sqrt(static_cast<double>(A.rows())));
    }
}

TEST_F(Blas, RejectsCsrInsteadOfSilentlySelectingNativeKernels)
{
    const cg::CsrMatrix A(2, 2, {0, 1, 2}, {0, 1}, {2, 3});
    const std::vector<double> b({1, 2}), x({0, 0});
    std::vector<double> output({7, 8});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend(1, true, cg::DenseKernel::blas);
    EXPECT_THROW(backend.apply(A, x, output), std::invalid_argument);
    EXPECT_THROW(backend.apply_and_dot(A, x, output), std::invalid_argument);
    EXPECT_THROW(backend.residual_squared_norm(A, b, x, output), std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A, b, output, workspace, backend), std::invalid_argument);
    EXPECT_DOUBLE_EQ(output.data()[0], 7);
    EXPECT_DOUBLE_EQ(output.data()[1], 8);
}

} // namespace
