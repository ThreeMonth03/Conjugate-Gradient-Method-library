#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/cg.hpp>
#include <cg/cuda.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>
#include <cg/solver/cuda/prepared_solver.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace
{
namespace gpu = cg::cuda;

class Cuda : public testing::Test
{
    protected:
    void SetUp() override
    {
        if (!gpu::available())
        {
            GTEST_SKIP() << "CUDA device is unavailable";
        }
        context = std::make_unique<gpu::Context>();
    }
    std::unique_ptr<gpu::Context> context;
};

cg::DenseMatrix system_matrix(std::size_t n)
{
    cg::DenseMatrix A(n, n);
    for (std::size_t i = 0; i < n; ++i)
    {
        A.data()[i * n + i] = 2.0 + static_cast<double>(i % 5);
        if (i + 1 < n)
        {
            A.data()[i * n + i + 1] = -0.25;
            A.data()[(i + 1) * n + i] = -0.25;
        }
    }
    return A;
}
cg::CsrMatrix as_csr(cg::DenseMatrixView A)
{
    std::vector<std::size_t> indptr{0}, indices;
    std::vector<double> values;
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        for (std::size_t column = 0; column < A.cols(); ++column)
        {
            const double value = A.data()[row * A.row_stride() + column];
            if (value != 0)
            {
                indices.push_back(column);
                values.push_back(value);
            }
        }
        indptr.push_back(indices.size());
    }
    return {A.rows(), A.cols(), std::move(indptr), std::move(indices), std::move(values)};
}
double true_norm(cg::DenseMatrixView A, cg::ConstVectorView b, cg::ConstVectorView x)
{
    double norm = 0.0;
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        long double sum = 0;
        for (std::size_t column = 0; column < A.cols(); ++column)
        {
            sum += static_cast<long double>(A.data()[row * A.row_stride() + column]) * x[column];
        }
        norm = std::hypot(norm, static_cast<double>(static_cast<long double>(b[row]) - sum));
    }
    return norm;
}
template <class T> struct DeviceIndices
{
    explicit DeviceIndices(const std::vector<T> &host)
        : storage(gpu::detail::device_allocation<T>(host.size(), gpu::Context{}))
    {
        if (cudaMemcpy(storage.get(), host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess)
        {
            throw std::runtime_error("test cudaMemcpy failed");
        }
    }
    gpu::detail::DeviceAllocation<T> storage;
};

TEST(CudaCapability, DeviceCountIsSafeWithoutAnAvailableDriver)
{
    EXPECT_GE(gpu::device_count(), 0);
    EXPECT_EQ(gpu::available(), gpu::device_count() > 0);
}

TEST_F(Cuda, ReportsVersionsAndRejectsInvalidContextAndKernel)
{
    EXPECT_GT(gpu::runtime_version(), 0);
    EXPECT_GT(gpu::driver_version(), 0);
    const auto info = gpu::device_info(0);
    EXPECT_FALSE(info.name.empty());
    EXPECT_GT(info.total_memory, 0u);
    EXPECT_THROW(gpu::Context(-1), std::invalid_argument);
    EXPECT_THROW(gpu::Context{gpu::device_count()}, std::invalid_argument);
    gpu::detail::ExecutionResources resources(*context);
    EXPECT_THROW(gpu::detail::Backend(resources, true, static_cast<gpu::DenseKernel>(-1)), std::invalid_argument);
}

TEST_F(Cuda, OwnsUploadsAndDeviceSnapshotsAndSupportsDeviceOnlyReset)
{
    std::vector<double> host({1, 2, 3}), downloaded(3);
    gpu::Vector source(host, *context), output(3, *context);
    host[0] = 99;
    output.copy_from_device(source.view(), *context);
    output.download(downloaded, *context);
    EXPECT_DOUBLE_EQ(downloaded[0], 1);
    output.fill(-2, *context);
    output.download(downloaded, *context);
    EXPECT_DOUBLE_EQ(downloaded[2], -2);
    output.copy_from_device(source.view(), *context);
    output.download(downloaded, *context);
    EXPECT_DOUBLE_EQ(downloaded[2], 3);
    gpu::Vector moved(std::move(output));
    EXPECT_EQ(output.size(), 0u);
    EXPECT_EQ(output.data(), nullptr);
    EXPECT_EQ(output.device(), -1);
    EXPECT_EQ(moved.size(), 3u);
    EXPECT_EQ(moved.device(), context->device());
    auto &same = moved;
    moved = std::move(same);
    EXPECT_EQ(moved.size(), 3u);
    output = std::move(moved);
    EXPECT_EQ(moved.size(), 0u);
    EXPECT_EQ(moved.device(), -1);
    output.download(downloaded, *context);
    EXPECT_DOUBLE_EQ(downloaded[2], 3);
    EXPECT_THROW(source.copy_from_device({source.data() + 1, 3, 0}, *context), std::invalid_argument);
    gpu::Vector empty(0, *context);
    EXPECT_NO_THROW(empty.fill(0, *context));
    EXPECT_NO_THROW(empty.download({}, *context));
}

TEST_F(Cuda, DenseProductsHandlePaddingOddTailsAndBothResidualSigns)
{
    const std::vector<double> storage{4, 1, 0, NAN, 1, 5, -1, NAN, 0, -1, 3, NAN};
    const gpu::DenseMatrix A(cg::DenseMatrixView(storage.data(), 3, 3, 4), *context);
    const std::vector<double> host_p({2, -1, 3}), host_b({8, -2, 11});
    gpu::Vector p(host_p, *context), b(host_b, *context), output(3, *context);
    std::vector<double> host_output(3);
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        gpu::detail::ExecutionResources resources(*context, kernel);

        for (const bool fused : {false, true})
        {
            const gpu::detail::Backend backend(resources, fused, kernel);
            output.fill(NAN, *context);
            EXPECT_DOUBLE_EQ(backend.apply_and_dot(A.view(), p.view(), output.view()), 50);
            output.download(host_output, *context);
            EXPECT_DOUBLE_EQ(host_output[0], 7);
            EXPECT_DOUBLE_EQ(host_output[1], -6);
            EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b.view(), p.view(), output.view()), 18);
            output.download(host_output, *context);
            EXPECT_DOUBLE_EQ(host_output[1], 4);
            EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A.view(), b.view(), p.view(), output.view(),
                                                           cg::ResidualSign::product_minus_rhs),
                             18);
            output.download(host_output, *context);
            EXPECT_DOUBLE_EQ(host_output[1], -4);
        }
    }
    const std::vector<double> raw_host({4, 1, 1, 3}), two_host({1, 2});
    gpu::Vector raw(raw_host, *context), two(two_host, *context), out(2, *context);
    const gpu::DenseMatrix snapshot({raw.data(), 2, 2, 2, 0}, *context);
    raw.fill(99, *context);
    gpu::detail::ExecutionResources resources(*context);
    const gpu::detail::Backend backend(resources);
    backend.apply(snapshot.view(), two.view(), out.view());
    std::vector<double> downloaded(2);
    out.download(downloaded, *context);
    EXPECT_DOUBLE_EQ(downloaded[0], 6);
    EXPECT_DOUBLE_EQ(downloaded[1], 7);
}

TEST_F(Cuda, RectangularAndEmptyProductsAvoidInvalidDeviceAccess)
{
    const cg::DenseMatrix host(2, 3, {1, 2, 3, -1, 0, 1});
    const gpu::DenseMatrix A(host.view(), *context);
    const std::vector<double> hx({2, 3, 4}), hb({1, -2});
    gpu::Vector x(hx, *context), b(hb, *context), out(2, *context);
    std::vector<double> result(2);

    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        gpu::detail::ExecutionResources resources(*context, kernel);

        const gpu::detail::Backend backend(resources, true, kernel);
        backend.apply(A.view(), x.view(), out.view());
        out.download(result, *context);
        EXPECT_DOUBLE_EQ(result[0], 20);
        EXPECT_DOUBLE_EQ(result[1], 2);
        const gpu::DenseMatrixView empty_columns{nullptr, 2, 0, 0, 0};
        backend.apply(empty_columns, {}, out.view());
        out.download(result, *context);
        EXPECT_DOUBLE_EQ(result[0], 0);
        EXPECT_DOUBLE_EQ(backend.residual_squared_norm(empty_columns, b.view(), {}, out.view()), 5);
        EXPECT_NO_THROW(backend.apply({nullptr, 0, 3, 3, 0}, x.view(), {}));
        EXPECT_DOUBLE_EQ(backend.apply_and_dot({nullptr, 0, 0, 0, 0}, {}, {}), 0);
    }
}

TEST_F(Cuda, RejectsShapePointerDeviceWorkspaceAndStorageAliasing)
{
    const cg::DenseMatrix host(2, 2, {2, 0, 0, 3});
    const gpu::DenseMatrix A(host.view(), *context);
    const std::vector<double> hb({1, 2});
    gpu::Vector b(hb, *context), x(2, *context);
    x.fill(0, *context);
    gpu::Workspace workspace(2, *context), wrong(3, *context);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), b.view(), workspace, *context), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), wrong, *context), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), {x.data(), 1, 0}, workspace, *context), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), {hb.data(), 2, 0}, x.view(), workspace, *context), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), {b.data(), 2, 1}, x.view(), workspace, *context), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), workspace.residual(), workspace, *context), std::invalid_argument);
    cg::CgOptions options;
    options.rtol = -1;
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options), std::invalid_argument);
    options.rtol = 1e-8;
    options.algorithm = static_cast<cg::Algorithm>(-1);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options), std::invalid_argument);
    options.algorithm = cg::Algorithm::recurrence;
    {
        gpu::detail::WorkspaceUseGuard lease(workspace, *context);
        EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options),
                     std::invalid_argument);
        lease.complete();
    }
    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options).status,
              cg::Status::converged);
}

TEST_F(Cuda, WorkspaceKernelSelectionIsPreparedBeforeSolving)
{
    const cg::DenseMatrix host(2, 2, {2, 0, 0, 3});
    const gpu::DenseMatrix A(host.view(), *context);
    const std::vector<double> rhs{1, 2};
    gpu::Vector b(rhs, *context), x(2, *context);
    gpu::Workspace native(2, *context), blas(2, *context, gpu::DenseKernel::cublas);
    EXPECT_EQ(native.dense_kernel(), gpu::DenseKernel::native);
    EXPECT_EQ(blas.dense_kernel(), gpu::DenseKernel::cublas);
    EXPECT_THROW(gpu::Workspace(2, *context, static_cast<gpu::DenseKernel>(-1)), std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), native, *context, {}, true, gpu::DenseKernel::cublas),
                 std::invalid_argument);
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), blas, *context), std::invalid_argument);
    // Rejected calls release the workspace; a subsequent correct solve is safe.
    x.fill(0, *context);
    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), native, *context).status, cg::Status::converged);
    x.fill(0, *context);
    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), blas, *context, {}, true, gpu::DenseKernel::cublas).status,
              cg::Status::converged);
}

TEST_F(Cuda, DenseCgAndPcgVerifyTrueResidualAcrossKernelsFusionAndRefresh)
{
    for (const std::size_t size : {17u, 257u})
    {
        const auto host = system_matrix(size);
        const gpu::DenseMatrix A(host.view(), *context);
        const gpu::JacobiPreconditioner M(A.view(), *context);
        const std::vector<double> hb(size, 1), initial(size, 0.125);
        gpu::Vector b(hb, *context), x(initial, *context);

        for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
        {
            gpu::Workspace selected_workspace(size, *context, kernel);
            const auto *allocation = selected_workspace.residual().data();

            for (const bool fused : {false, true})
            {
                for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
                {
                    for (const bool preconditioned : {false, true})
                    {
                        if (preconditioned && algorithm != cg::Algorithm::recurrence)
                        {
                            continue;
                        }
                        x.upload(initial, *context);
                        cg::CgOptions options;
                        options.algorithm = algorithm;
                        options.rtol = 1e-9;
                        options.maxiter = 300;
                        options.residual_refresh = 7;
                        const auto info = gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, *context,
                                                          options, fused, kernel, preconditioned ? &M : nullptr);
                        ASSERT_EQ(info.status, cg::Status::converged);
                        std::vector<double> result(size);
                        x.download(result, *context);
                        EXPECT_LE(true_norm(host.view(), hb, result),
                                  options.rtol * std::sqrt(static_cast<double>(size)));
                        EXPECT_EQ(selected_workspace.residual().data(), allocation);
                        const auto repeated =
                            gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, *context, options, fused,
                                            kernel, preconditioned ? &M : nullptr);
                        EXPECT_EQ(repeated.iterations, 0u);
                        EXPECT_EQ(repeated.matvecs, 1u);
                    }
                }
            }
        }
    }
}

TEST_F(Cuda, CsrCgAndPcgMatchDenseAndRejectCublas)
{
    const auto host = system_matrix(33);
    const gpu::CsrMatrix A(as_csr(host.view()), *context);
    const gpu::JacobiPreconditioner M(A, *context);
    const std::vector<double> hb(33, 1);
    gpu::Vector b(hb, *context), x(33, *context);
    gpu::Workspace workspace(33, *context);
    for (const bool fused : {false, true})
    {
        for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
        {
            for (const bool preconditioned : {false, true})
            {
                if (preconditioned && algorithm != cg::Algorithm::recurrence)
                {
                    continue;
                }
                x.fill(0, *context);
                cg::CgOptions options;
                options.algorithm = algorithm;
                options.residual_refresh = 7;
                options.rtol = 1e-9;
                const auto info = gpu::solve_into(A, b.view(), x.view(), workspace, *context, options, fused,
                                                  gpu::DenseKernel::native, preconditioned ? &M : nullptr);
                ASSERT_EQ(info.status, cg::Status::converged);
                std::vector<double> result(33);
                x.download(result, *context);
                EXPECT_LE(true_norm(host.view(), hb, result), options.rtol * std::sqrt(33.0));
            }
        }
    }
    EXPECT_THROW(gpu::solve_into(A, b.view(), x.view(), workspace, *context, {}, true, gpu::DenseKernel::cublas),
                 std::invalid_argument);
    gpu::detail::ExecutionResources cublas_resources(*context, gpu::DenseKernel::cublas);
    const gpu::detail::Backend backend(cublas_resources, true, gpu::DenseKernel::cublas);
    EXPECT_THROW(backend.apply(A, b.view(), x.view()), std::invalid_argument);
}

TEST_F(Cuda, NativePreparedPlansCacheSetupAndReuseIndependentWorkspaces)
{
    const auto host = system_matrix(17);
    const std::vector<double> rhs(17, 1.0);
    const gpu::Vector b(rhs, *context);
    const auto check = [&](const gpu::PreparedSolver &plan, cg::PreconditionerKind kind)
    {
        EXPECT_EQ(plan.size(), 17u);
        EXPECT_EQ(plan.device(), context->device());
        EXPECT_EQ(plan.preconditioner(), kind);
        EXPECT_EQ(plan.preconditioner_nbytes(),
                  kind == cg::PreconditionerKind::jacobi ? rhs.size() * sizeof(double) : 0u);
        if (kind == cg::PreconditionerKind::none)
        {
            EXPECT_EQ(plan.preconditioner_setup_ns(), 0u);
        }
        auto workspace = plan.create_workspace();
        auto independent = plan.create_workspace();
        EXPECT_EQ(workspace.dense_kernel(), plan.dense_kernel());
        EXPECT_NE(workspace.residual().data(), independent.residual().data());
        gpu::Vector x(17, *context);
        x.fill(0, *context);
        cg::CgOptions options;
        options.rtol = 1e-10;
        ASSERT_EQ(plan.solve_into(b.view(), x.view(), workspace, options).status, cg::Status::converged);
        std::vector<double> solution(17);
        x.download(solution, *context);
        EXPECT_LE(true_norm(host.view(), rhs, solution), options.rtol * std::sqrt(17.0));
        const auto repeated = plan.solve_into(b.view(), x.view(), workspace, options);
        EXPECT_EQ(repeated.iterations, 0u);
        EXPECT_EQ(repeated.matvecs, 1u);
        x.fill(0, *context);
        ASSERT_EQ(plan.solve_into(b.view(), x.view(), independent, options).status, cg::Status::converged);
    };
    for (const auto kind : {cg::PreconditionerKind::none, cg::PreconditionerKind::jacobi})
    {
        for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
        {
            const gpu::PreparedSolver plan(gpu::DenseMatrix(host.view(), *context), *context, kernel, kind);
            check(plan, kind);
        }
        const gpu::PreparedSolver sparse(gpu::CsrMatrix(as_csr(host.view()), *context), *context,
                                         gpu::DenseKernel::native, kind);
        check(sparse, kind);
    }
    EXPECT_THROW(gpu::PreparedSolver(gpu::CsrMatrix(as_csr(host.view()), *context), *context, gpu::DenseKernel::cublas),
                 std::invalid_argument);
}

TEST_F(Cuda, OneShotPreconditionerPolicyRejectsInvalidOptionsBeforeSetup)
{
    const cg::DenseMatrix host(2, 2, {1, 0, 0, -1});
    const gpu::DenseMatrix A(host.view(), *context);
    const std::vector<double> rhs{1, 1};
    const gpu::Vector b(rhs, *context);
    gpu::Vector x(2, *context);
    x.fill(0, *context);
    gpu::Workspace workspace(2, *context);
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    try
    {
        gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options, true, gpu::DenseKernel::native,
                        cg::PreconditionerKind::jacobi);
        FAIL() << "explicit CG must reject Jacobi before reading its diagonal";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_STREQ(error.what(), "Jacobi preconditioning requires recurrence CG");
    }
    EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, {}, true, gpu::DenseKernel::native,
                                 static_cast<cg::PreconditionerKind>(-1)),
                 std::invalid_argument);
    std::vector<double> output(2);
    x.download(output, *context);
    EXPECT_EQ(output, (std::vector<double>{0, 0}));
}

TEST_F(Cuda, CsrDeviceSnapshotsValidateCanonicalInt32AndInt64Indices)
{
    const std::vector<double> values({2, 3}), hb({1, 1});
    gpu::Vector data(values, *context), b(hb, *context), x(2, *context);
    DeviceIndices<std::int32_t> ptr32({0, 1, 2}), idx32({0, 1}), bad32({0, 0});
    DeviceIndices<std::int64_t> ptr64({0, 1, 2}), idx64({0, 1});
    const gpu::CsrMatrixView views[] = {
        {ptr32.storage.get(), idx32.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int32, 0},
        {ptr64.storage.get(), idx64.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int64, 0}};
    for (const auto view : views)
    {
        const gpu::CsrMatrix A(view, *context);
        x.fill(0, *context);
        gpu::Workspace workspace(2, *context);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        ASSERT_EQ(gpu::solve_into(A, b.view(), x.view(), workspace, *context, options).status, cg::Status::converged);
        std::vector<double> result(2);
        x.download(result, *context);
        EXPECT_NEAR(result[0], 0.5, 1e-14);
        EXPECT_NEAR(result[1], 1.0 / 3.0, 1e-14);
    }
    DeviceIndices<std::int32_t> duplicate_ptr({0, 2, 2});
    const gpu::CsrMatrixView bad{
        duplicate_ptr.storage.get(), bad32.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int32, 0};
    EXPECT_THROW(gpu::CsrMatrix(bad, *context), std::invalid_argument);
    DeviceIndices<std::int32_t> negative32({0, -1}), out_of_range32({0, 2});
    DeviceIndices<std::int64_t> negative64({0, -1});
    const gpu::CsrMatrixView invalid[] = {
        {ptr32.storage.get(), negative32.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int32, 0},
        {ptr64.storage.get(), negative64.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int64, 0},
        {ptr32.storage.get(), out_of_range32.storage.get(), data.data(), 2, 2, 2, gpu::IndexType::int32, 0},
    };
    for (const auto view : invalid)
    {
        EXPECT_THROW(gpu::CsrMatrix(view, *context), std::invalid_argument);
    }
}

TEST_F(Cuda, JacobiRejectsInvalidMissingAndUnrepresentableDiagonals)
{
    for (const double value : {0.0, -1.0, std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(), std::numeric_limits<double>::denorm_min()})
    {
        const cg::DenseMatrix host(2, 2, {1, 0, 0, value});
        const gpu::DenseMatrix A(host.view(), *context);
        EXPECT_THROW(gpu::JacobiPreconditioner(A.view(), *context), std::invalid_argument);
        const cg::CsrMatrix hs(2, 2, {0, 1, 2}, {0, 1}, {1, value});
        const gpu::CsrMatrix sparse(hs, *context);
        EXPECT_THROW(gpu::JacobiPreconditioner(sparse, *context), std::invalid_argument);
    }
    const gpu::CsrMatrix missing(cg::CsrMatrix(2, 2, {0, 1, 1}, {0}, {1}), *context);
    EXPECT_THROW(gpu::JacobiPreconditioner(missing, *context), std::invalid_argument);
}

TEST_F(Cuda, StableNormAndRangeFailuresNeverClaimFalseConvergence)
{
    const cg::DenseMatrix host(2, 2, {1, 0, 0, 1});
    const gpu::DenseMatrix A(host.view(), *context);

    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        gpu::Workspace selected_workspace(2, *context, kernel);

        for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
        {
            for (const double magnitude : {1e-200, 1e-160, 1e200})
            {
                const std::vector<double> hb({magnitude, -magnitude});
                gpu::Vector b(hb, *context), x(2, *context);
                x.fill(0, *context);
                cg::CgOptions options;
                options.algorithm = algorithm;
                const auto info =
                    gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, *context, options, true, kernel);
                EXPECT_EQ(info.status, cg::Status::non_finite);
                EXPECT_EQ(info.iterations, 0u);
                EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
                options.maxiter = 0;
                EXPECT_EQ(
                    gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, *context, options, true, kernel)
                        .status,
                    cg::Status::max_iterations);
                if (magnitude == 1e-200)
                {
                    options.atol = 1e-199;
                    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, *context, options, true,
                                              kernel)
                                  .status,
                              cg::Status::converged);
                }
            }
        }
    }
}

TEST_F(Cuda, NonfiniteProductsAndNegativeCurvatureReportNumericalStatuses)
{
    const std::vector<double> hb({1, 2});
    gpu::Vector b(hb, *context), x(2, *context);

    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        gpu::Workspace selected_workspace(2, *context, kernel);

        for (const auto algorithm : {cg::Algorithm::explicit_residual, cg::Algorithm::recurrence})
        {
            cg::CgOptions options;
            options.algorithm = algorithm;
            const gpu::DenseMatrix negative(cg::DenseMatrix(2, 2, {-1, 0, 0, -1}).view(), *context);
            x.fill(0, *context);
            EXPECT_EQ(gpu::solve_into(negative.view(), b.view(), x.view(), selected_workspace, *context, options, true,
                                      kernel)
                          .status,
                      cg::Status::non_positive_curvature);
            const gpu::DenseMatrix invalid(cg::DenseMatrix(2, 2, {1, INFINITY, 0, 1}).view(), *context);
            x.fill(0, *context);
            EXPECT_EQ(
                gpu::solve_into(invalid.view(), b.view(), x.view(), selected_workspace, *context, options, true, kernel)
                    .status,
                cg::Status::non_finite);
        }
    }
}

TEST_F(Cuda, NondefaultStreamIsBorrowedAndWorkspacesCannotCrossStreams)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        const gpu::Context selected(0, reinterpret_cast<std::uintptr_t>(stream));
        const auto host = system_matrix(17);
        const gpu::DenseMatrix A(host.view(), selected);
        const std::vector<double> hb(17, 1);
        gpu::Vector b(hb, selected), x(17, selected);
        x.fill(0, selected);
        gpu::Workspace workspace(17, selected);
        EXPECT_EQ(workspace.stream(), reinterpret_cast<std::uintptr_t>(stream));
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        ASSERT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), workspace, selected, options).status,
                  cg::Status::converged);
        EXPECT_THROW(gpu::solve_into(A.view(), b.view(), x.view(), workspace, *context, options),
                     std::invalid_argument);
        unsigned int flags = 0;
        EXPECT_EQ(cudaStreamGetFlags(stream, &flags), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(Cuda, ImmutablePreparedPlanSupportsConcurrentSolvesOnIndependentStreams)
{
    const auto host = system_matrix(33);
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        const gpu::PreparedSolver plan(gpu::DenseMatrix(host.view(), *context), *context, kernel,
                                       cg::PreconditionerKind::jacobi);
        const auto run = [&](double rhs)
        {
            cudaStream_t stream = nullptr;
            if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
            {
                throw std::runtime_error("test stream creation failed");
            }
            double residual = 0;
            cg::Status status = cg::Status::max_iterations;
            {
                const auto stream_address = reinterpret_cast<std::uintptr_t>(stream);
                const gpu::Context selected(0, stream_address);
                const std::vector<double> hb(33, rhs);
                gpu::Vector b(hb, selected), x(33, selected);
                x.fill(0, selected);
                auto workspace = plan.create_workspace(stream_address);
                cg::CgOptions options;
                options.algorithm = cg::Algorithm::recurrence;
                options.rtol = 1e-9;
                status = plan.solve_into(b.view(), x.view(), workspace, options).status;
                std::vector<double> downloaded(33);
                x.download(downloaded, selected);
                residual = true_norm(host.view(), hb, downloaded);
            }
            (void)cudaStreamDestroy(stream);
            return std::make_pair(status, residual);
        };
        auto first = std::async(std::launch::async, run, 1.0), second = std::async(std::launch::async, run, -2.0);
        for (auto *future : {&first, &second})
        {
            const auto result = future->get();
            EXPECT_EQ(result.first, cg::Status::converged);
            EXPECT_LE(result.second, 2e-9 * std::sqrt(33.0));
        }
    }
}

} // namespace
