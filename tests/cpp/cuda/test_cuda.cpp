#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/backend/cuda/detail/execution_resources.hpp>
#include <cg/cg.hpp>
#include <cg/cuda.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>
#include <cg/solver/cuda/prepared_solver.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
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

static_assert(!std::is_copy_constructible_v<gpu::CsrMatrix>);
static_assert(!std::is_copy_assignable_v<gpu::CsrMatrix>);
static_assert(std::is_nothrow_move_constructible_v<gpu::CsrMatrix>);
static_assert(std::is_nothrow_move_assignable_v<gpu::CsrMatrix>);
static_assert(!std::is_copy_constructible_v<gpu::JacobiPreconditioner>);
static_assert(!std::is_copy_assignable_v<gpu::JacobiPreconditioner>);
static_assert(std::is_nothrow_move_constructible_v<gpu::JacobiPreconditioner>);
static_assert(std::is_nothrow_move_assignable_v<gpu::JacobiPreconditioner>);
static_assert(!std::is_move_constructible_v<gpu::detail::ExecutionResources>);
static_assert(!std::is_move_assignable_v<gpu::detail::ExecutionResources>);

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
    EXPECT_THROW(gpu::detail::ExecutionResources(*context, static_cast<gpu::DenseKernel>(-1)), std::invalid_argument);
}

TEST_F(Cuda, DeviceGuardsPreserveSameDeviceAcrossNestingAndExceptionUnwind)
{
    int original = -1;
    ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
    const auto expect_device = [](int expected)
    {
        int selected = -1;
        EXPECT_EQ(cudaGetDevice(&selected), cudaSuccess);
        EXPECT_EQ(selected, expected);
    };
    struct Unwind
    {
    };
    {
        const gpu::detail::DeviceGuard outer(original);
        expect_device(original);
        {
            const gpu::detail::DeviceGuard inner(original);
            expect_device(original);
        }
        expect_device(original);
        EXPECT_THROW(
            {
                const gpu::detail::DeviceGuard unwinding(original);
                expect_device(original);
                throw Unwind{};
            },
            Unwind);
        expect_device(original);
    }
    expect_device(original);
}

TEST_F(Cuda, DeviceGuardsRestoreSwitchedAndCallbackChangedDevicesAcrossUnwind)
{
    const int count = gpu::device_count();
    if (count < 2)
    {
        GTEST_SKIP() << "Device switching and callback restoration require at least two CUDA devices";
    }
    int original = -1;
    ASSERT_EQ(cudaGetDevice(&original), cudaSuccess);
    const int other = (original + 1) % count;
    const auto expect_device = [](int expected)
    {
        int selected = -1;
        EXPECT_EQ(cudaGetDevice(&selected), cudaSuccess);
        EXPECT_EQ(selected, expected);
    };
    struct Unwind
    {
    };
    {
        const gpu::detail::DeviceGuard outer(other);
        expect_device(other);
        {
            const gpu::detail::DeviceGuard inner(original);
            expect_device(original);
        }
        expect_device(other);
        EXPECT_THROW(
            {
                const gpu::detail::DeviceGuard unwinding(original);
                expect_device(original);
                throw Unwind{};
            },
            Unwind);
        expect_device(other);
        {
            const gpu::detail::DeviceGuard same(other);
            // CUDA Array Interface metadata callbacks can change selection
            // inside a guard that needed no device switch at construction.
            ASSERT_EQ(cudaSetDevice(original), cudaSuccess);
        }
        expect_device(other);
        EXPECT_THROW(
            {
                const gpu::detail::DeviceGuard unwinding(other);
                ASSERT_EQ(cudaSetDevice(original), cudaSuccess);
                throw Unwind{};
            },
            Unwind);
        expect_device(other);
    }
    expect_device(original);
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
    const gpu::Vector empty_upload(cg::ConstVectorView{}, *context);
    EXPECT_EQ(empty_upload.size(), 0u);
    EXPECT_EQ(empty_upload.device(), context->device());
    EXPECT_NO_THROW(empty_upload.download({}, *context));
}

TEST_F(Cuda, ConstructorsRejectWrongDeviceBeforeAttemptingOversizedAllocation)
{
    // The byte count is representable, but no device can allocate this span.
    // Metadata errors must take precedence over a cudaMalloc failure.
    const auto size = std::numeric_limits<std::size_t>::max() / sizeof(double) - 1;
    const gpu::DenseMatrixView wrong_device(nullptr, size, 1, 1, context->device() + 1);
    try
    {
        const gpu::DenseMatrix snapshot(wrong_device, *context);
        FAIL() << "wrong-device matrix storage must be rejected before allocation";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_STREQ(error.what(), "CUDA storage belongs to a different device");
    }
    try
    {
        const gpu::JacobiPreconditioner M(wrong_device, *context);
        FAIL() << "wrong-device Jacobi input must be rejected before allocation";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_STREQ(error.what(), "CUDA storage belongs to a different device");
    }
}

TEST_F(Cuda, DenseSnapshotsAndJacobiValidateDescriptorsAndShapes)
{
    const cg::DenseMatrix host(2, 3, {1, 2, 3, 4, 5, 6});
    const gpu::DenseMatrix A(host.view(), *context);
    const gpu::CsrMatrix sparse(as_csr(host.view()), *context);
    EXPECT_THROW(gpu::DenseMatrix(gpu::DenseMatrixView{nullptr, 2, 3, 3, context->device()}, *context),
                 std::invalid_argument);
    EXPECT_THROW(gpu::DenseMatrix(gpu::DenseMatrixView{A.view().data(), 2, 3, 2, context->device()}, *context),
                 std::invalid_argument);
    EXPECT_THROW(gpu::JacobiPreconditioner(A.view(), *context), std::invalid_argument);
    EXPECT_THROW(gpu::JacobiPreconditioner(sparse, *context), std::invalid_argument);

    const gpu::DenseMatrix empty_host(cg::DenseMatrixView{}, *context);
    const gpu::DenseMatrix empty_snapshot(gpu::DenseMatrixView{nullptr, 0, 3, 3, -1}, *context);
    EXPECT_EQ(empty_host.rows(), 0u);
    EXPECT_EQ(empty_host.cols(), 0u);
    EXPECT_EQ(empty_snapshot.rows(), 0u);
    EXPECT_EQ(empty_snapshot.cols(), 3u);
    EXPECT_THROW(gpu::JacobiPreconditioner(empty_host.view(), *context), std::invalid_argument);
    const gpu::CsrMatrix empty_csr(cg::CsrMatrix(0, 0, {0}, {}, {}), *context);
    EXPECT_THROW(gpu::JacobiPreconditioner(empty_csr, *context), std::invalid_argument);
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
            const gpu::detail::Backend backend(resources, fused);
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

TEST_F(Cuda, ExecutionStorageKeepsProviderScratchDisjointAndReductionReusable)
{
    constexpr std::size_t size = 257, stride = size + 3;
    std::vector<double> host_matrix(size * stride, NAN), host_p(size), host_b(size), expected_product(size);
    double expected_dot = 0.0, expected_p_squared = 0.0;
    for (std::size_t row = 0; row < size; ++row)
    {
        std::fill_n(host_matrix.data() + row * stride, size, 0.0);
        const double diagonal = 1.0 + row % 4;
        host_matrix[row * stride + row] = diagonal;
        host_p[row] = static_cast<int>(row % 5) - 2;
        expected_product[row] = diagonal * host_p[row];
        host_b[row] = expected_product[row] + 1.0;
        expected_dot += host_p[row] * expected_product[row];
        expected_p_squared += host_p[row] * host_p[row];
    }
    // Keep device padding: a DenseMatrix snapshot would pack the active entries.
    const gpu::Vector matrix_storage(host_matrix, *context);
    const gpu::DenseMatrixView A(matrix_storage.data(), size, size, stride, context->device());
    gpu::Vector p(host_p, *context), b(host_b, *context), output(size, *context);
    std::vector<double> actual(size);
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        SCOPED_TRACE(static_cast<int>(kernel));
        gpu::detail::ExecutionResources resources(*context, kernel);
        const auto base = reinterpret_cast<std::uintptr_t>(resources.storage.get());
        const auto partials = reinterpret_cast<std::uintptr_t>(resources.partials);
        const auto scalar = reinterpret_cast<std::uintptr_t>(resources.scalar);
        EXPECT_EQ(base % 256, 0u);
        const std::size_t scratch_bytes =
            kernel == gpu::DenseKernel::cublas ? gpu::detail::CublasResources::workspace_bytes : 0;
        EXPECT_EQ(partials - base, scratch_bytes);
        EXPECT_EQ(partials % alignof(double), 0u);
        EXPECT_EQ(scalar - partials, gpu::detail::max_blocks * sizeof(double));
        EXPECT_EQ(resources.dense_kernel(), kernel);
        if (kernel == gpu::DenseKernel::cublas)
        {
            ASSERT_TRUE(resources.cublas.has_value());
            EXPECT_EQ(resources.cublas->workspace.data(), resources.storage.get());
            EXPECT_EQ(resources.cublas->workspace.size_bytes(), scratch_bytes);
            EXPECT_EQ(reinterpret_cast<std::uintptr_t>(resources.cublas->workspace.data()) +
                          resources.cublas->workspace.size_bytes(),
                      partials);
        }
        else
        {
            EXPECT_FALSE(resources.cublas.has_value());
        }
        for (const bool fused : {false, true})
        {
            SCOPED_TRACE(fused);
            const gpu::detail::Backend backend(resources, fused);
            for (int repeat = 0; repeat < 3; ++repeat)
            {
                EXPECT_DOUBLE_EQ(backend.apply_and_dot(A, p.view(), output.view()), expected_dot);
                output.download(actual, *context);
                EXPECT_EQ(actual, expected_product);
                EXPECT_DOUBLE_EQ(backend.dot(p.view(), p.view()), expected_p_squared);
                EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A, b.view(), p.view(), output.view()), size);
                output.download(actual, *context);
                EXPECT_EQ(actual, std::vector<double>(size, 1.0));
                EXPECT_DOUBLE_EQ(backend.norm(output.view()), std::sqrt(static_cast<double>(size)));
            }
        }
    }
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

        const gpu::detail::Backend backend(resources, true);
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
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), b.view(), workspace)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), x.view(), wrong)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), {x.data(), 1, 0}, workspace)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), {hb.data(), 2, 0}, x.view(), workspace)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), {b.data(), 2, 1}, x.view(), workspace)),
                 std::invalid_argument);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), workspace.residual(), workspace)),
                 std::invalid_argument);
    cg::CgOptions options;
    options.rtol = -1;
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), x.view(), workspace, options)),
                 std::invalid_argument);
    options.rtol = 1e-8;
    options.algorithm = static_cast<cg::Algorithm>(-1);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), x.view(), workspace, options)),
                 std::invalid_argument);
    options.algorithm = cg::Algorithm::recurrence;
    {
        gpu::detail::WorkspaceUseGuard lease(workspace);
        EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), x.view(), workspace, options)),
                     std::invalid_argument);
        lease.complete();
    }
    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), workspace, options).status, cg::Status::converged);
}

TEST_F(Cuda, WorkspaceOwnsExecutionSettingsAndRejectsIncompatiblePlansBeforeWrites)
{
    const cg::DenseMatrix host(2, 2, {2, 0, 0, 3});
    const gpu::DenseMatrix A(host.view(), *context);
    const std::vector<double> rhs{1, 2};
    gpu::Vector b(rhs, *context), x(2, *context);
    EXPECT_THROW(gpu::Workspace(2, *context, static_cast<gpu::DenseKernel>(-1)), std::invalid_argument);
    for (const auto mode : {gpu::ExecutionMode::host, gpu::ExecutionMode::device})
    {
        for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
        {
            gpu::Workspace workspace(2, *context, kernel, mode);
            EXPECT_EQ(workspace.context().device(), context->device());
            EXPECT_EQ(workspace.context().stream(), context->stream());
            EXPECT_EQ(workspace.dense_kernel(), kernel);
            EXPECT_EQ(workspace.execution_mode(), mode);
            x.fill(0, *context);
            EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), workspace).status, cg::Status::converged);
            std::vector<double> actual(2);
            x.download(actual, *context);
            EXPECT_NEAR(actual[0], 0.5, 1e-14);
            EXPECT_NEAR(actual[1], 2.0 / 3.0, 1e-14);

            const auto other = kernel == gpu::DenseKernel::native ? gpu::DenseKernel::cublas : gpu::DenseKernel::native;
            const gpu::PreparedSolver plan(gpu::DenseMatrix(host.view(), *context), *context,
                                           cg::PreconditionerKind::none, kernel);
            gpu::Workspace mismatched(2, *context, other, mode);
            x.fill(-7, *context);
            EXPECT_THROW(static_cast<void>(plan.solve_into(b.view(), x.view(), mismatched)), std::invalid_argument);
            x.download(actual, *context);
            EXPECT_EQ(actual, (std::vector<double>{-7, -7}));
        }
    }
}

TEST_F(Cuda, WorkspaceRejectsCombinedElementAndByteSizeOverflow)
{
    constexpr auto limit = std::numeric_limits<std::size_t>::max();
    EXPECT_THROW(gpu::Workspace(limit / 5 + 1, *context), std::invalid_argument);
    EXPECT_THROW(gpu::Workspace(limit / (5 * sizeof(double)) + 1, *context), std::invalid_argument);
}

TEST_F(Cuda, WorkspaceBuffersRemainIndependentForEmptyAndOddSizes)
{
    for (const std::size_t size : {0u, 1u, 17u, 257u})
    {
        SCOPED_TRACE(size);
        gpu::Workspace workspace(size, *context);
        const std::array<gpu::VectorView, 5> buffers{workspace.residual(), workspace.direction(), workspace.product(),
                                                     workspace.scratch(), workspace.preconditioned()};
        std::vector<double> values(size);
        for (std::size_t index = 0; index < buffers.size(); ++index)
        {
            EXPECT_EQ(buffers[index].size(), size);
            EXPECT_EQ(buffers[index].device(), context->device());
            if (size == 0)
            {
                EXPECT_EQ(buffers[index].data(), nullptr);
                continue;
            }
            std::fill(values.begin(), values.end(), static_cast<double>(index + 1));
            ASSERT_EQ(cudaMemcpy(buffers[index].data(), values.data(), size * sizeof(double), cudaMemcpyHostToDevice),
                      cudaSuccess);
        }
        for (std::size_t index = 0; index < buffers.size() && size != 0; ++index)
        {
            ASSERT_EQ(cudaMemcpy(values.data(), buffers[index].data(), size * sizeof(double), cudaMemcpyDeviceToHost),
                      cudaSuccess);
            for (const double value : values)
            {
                EXPECT_DOUBLE_EQ(value, static_cast<double>(index + 1));
            }
        }
    }
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
                        const auto info = gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, options,
                                                          fused, preconditioned ? &M : nullptr);
                        ASSERT_EQ(info.status, cg::Status::converged);
                        std::vector<double> result(size);
                        x.download(result, *context);
                        EXPECT_LE(true_norm(host.view(), hb, result),
                                  options.rtol * std::sqrt(static_cast<double>(size)));
                        EXPECT_EQ(selected_workspace.residual().data(), allocation);
                        const auto repeated = gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, options,
                                                              fused, preconditioned ? &M : nullptr);
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
                const auto info =
                    gpu::solve_into(A, b.view(), x.view(), workspace, options, fused, preconditioned ? &M : nullptr);
                ASSERT_EQ(info.status, cg::Status::converged);
                std::vector<double> result(33);
                x.download(result, *context);
                EXPECT_LE(true_norm(host.view(), hb, result), options.rtol * std::sqrt(33.0));
            }
        }
    }
    gpu::Workspace cublas_workspace(33, *context, gpu::DenseKernel::cublas);
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A, b.view(), x.view(), cublas_workspace)), std::invalid_argument);
    gpu::detail::ExecutionResources cublas_resources(*context, gpu::DenseKernel::cublas);
    const gpu::detail::Backend backend(cublas_resources, true);
    EXPECT_THROW(backend.apply(A, b.view(), x.view()), std::invalid_argument);
}

TEST_F(Cuda, DeviceExecutionMatchesHostPoliciesCountsAndTrueResiduals)
{
    const auto host = system_matrix(17);
    const std::vector<double> rhs(17, 1.0), initial(17, 0.125);
    gpu::Vector b(rhs, *context), x_host(17, *context), x_device(17, *context);
    const auto check = [&](const auto &A, gpu::DenseKernel kernel)
    {
        const gpu::JacobiPreconditioner M(A, *context);
        gpu::Workspace host_work(17, *context, kernel);
        gpu::Workspace device_work(17, *context, kernel, gpu::ExecutionMode::device);
        EXPECT_EQ(host_work.execution_mode(), gpu::ExecutionMode::host);
        EXPECT_EQ(device_work.execution_mode(), gpu::ExecutionMode::device);
        for (const bool fused : {false, true})
        {
            SCOPED_TRACE(fused);
            for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
            {
                SCOPED_TRACE(static_cast<int>(algorithm));
                for (const bool preconditioned : {false, true})
                {
                    if (preconditioned && algorithm != cg::Algorithm::recurrence)
                    {
                        continue;
                    }
                    SCOPED_TRACE(preconditioned);
                    for (const std::size_t refresh : {0u, 1u, 7u})
                    {
                        SCOPED_TRACE(refresh);
                        for (const std::size_t maxiter : {0u, 1u, 80u})
                        {
                            SCOPED_TRACE(maxiter);
                            cg::CgOptions options;
                            options.algorithm = algorithm;
                            options.rtol = 1e-8;
                            options.maxiter = maxiter;
                            options.residual_refresh = refresh;
                            x_host.upload(initial, *context);
                            x_device.upload(initial, *context);
                            const auto *preconditioner = preconditioned ? &M : nullptr;
                            const auto reference =
                                gpu::solve_into(A, b.view(), x_host.view(), host_work, options, fused, preconditioner);
                            const auto result = gpu::solve_into(A, b.view(), x_device.view(), device_work, options,
                                                                fused, preconditioner);
                            EXPECT_EQ(result.status, reference.status);
                            EXPECT_EQ(result.iterations, reference.iterations);
                            EXPECT_EQ(result.matvecs, reference.matvecs);
                            EXPECT_EQ(result.restarts, reference.restarts);
                            EXPECT_NEAR(result.residual_norm, reference.residual_norm, 1e-12);
                            std::vector<double> expected(17), actual(17);
                            x_host.download(expected, *context);
                            x_device.download(actual, *context);
                            for (std::size_t i = 0; i < actual.size(); ++i)
                            {
                                EXPECT_NEAR(actual[i], expected[i], 1e-12);
                            }
                            EXPECT_NEAR(result.residual_norm, true_norm(host.view(), rhs, actual), 1e-12);
                            if (maxiter == 0)
                            {
                                EXPECT_EQ(result.matvecs, 1u);
                            }
                            if (refresh == 1 && algorithm == cg::Algorithm::recurrence && result.iterations > 1)
                            {
                                EXPECT_EQ(result.restarts, result.iterations - 1);
                                EXPECT_EQ(result.matvecs, 1 + 2 * result.iterations);
                            }
                        }
                    }
                }
            }
        }
    };
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        SCOPED_TRACE(static_cast<int>(kernel));
        const gpu::DenseMatrix A(host.view(), *context);
        check(A.view(), kernel);
    }
    const gpu::CsrMatrix sparse(as_csr(host.view()), *context);
    check(sparse, gpu::DenseKernel::native);
    EXPECT_THROW(gpu::Workspace(17, *context, gpu::DenseKernel::native, static_cast<gpu::ExecutionMode>(-1)),
                 std::invalid_argument);
}

TEST_F(Cuda, DeviceFailuresPreserveOutputCountsAndPermitWorkspaceReuse)
{
    const std::vector<double> rhs{1.0, 2.0}, initial{0.0, 0.0};
    gpu::Vector b(rhs, *context), x_host(2, *context), x_device(2, *context);
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        gpu::Workspace host_work(2, *context, kernel);
        gpu::Workspace device_work(2, *context, kernel, gpu::ExecutionMode::device);
        const cg::DenseMatrix host_good(2, 2, {2, 0, 0, 3});
        const gpu::DenseMatrix good(host_good.view(), *context);
        for (const bool fused : {false, true})
        {
            for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
            {
                cg::CgOptions options;
                options.algorithm = algorithm;
                for (const double diagonal : {-1.0, 0.0, 1e-310, 1e308})
                {
                    SCOPED_TRACE(diagonal);
                    const cg::DenseMatrix host(2, 2, {diagonal, 0, 0, diagonal});
                    const gpu::DenseMatrix A(host.view(), *context);
                    x_host.upload(initial, *context);
                    x_device.upload(initial, *context);
                    const auto reference =
                        gpu::solve_into(A.view(), b.view(), x_host.view(), host_work, options, fused);
                    const auto result =
                        gpu::solve_into(A.view(), b.view(), x_device.view(), device_work, options, fused);
                    EXPECT_EQ(result.status, reference.status);
                    EXPECT_EQ(result.iterations, 0u);
                    EXPECT_EQ(result.matvecs, reference.matvecs);
                    EXPECT_EQ(result.restarts, reference.restarts);
                    EXPECT_DOUBLE_EQ(result.residual_norm, reference.residual_norm);
                    std::vector<double> actual(2);
                    x_device.download(actual, *context);
                    EXPECT_EQ(actual, initial);
                    EXPECT_EQ(
                        gpu::solve_into(good.view(), b.view(), x_device.view(), device_work, options, fused).status,
                        cg::Status::converged);
                }
                const gpu::JacobiPreconditioner M(good.view(), *context);
                if (algorithm == cg::Algorithm::explicit_residual)
                {
                    x_device.upload(initial, *context);
                    EXPECT_THROW(static_cast<void>(gpu::solve_into(good.view(), b.view(), x_device.view(), device_work,
                                                                   options, fused, &M)),
                                 std::invalid_argument);
                    std::vector<double> actual(2);
                    x_device.download(actual, *context);
                    EXPECT_EQ(actual, initial);
                }
            }
        }
    }
}

TEST_F(Cuda, DevicePacketResetPreservesCompletedStepsAcrossBreakdownRestartAndReuse)
{
    // The first step has positive curvature; its next direction has negative
    // curvature. A stale advanced flag must not count the failed second step.
    const cg::DenseMatrix host(2, 2, {1, 0, 0, -1});
    const std::vector<double> rhs{2.0, 1.0};
    gpu::Vector b(rhs, *context), x_host(2, *context), x_device(2, *context);
    const auto check = [&](const auto &A, gpu::DenseKernel kernel)
    {
        gpu::Workspace host_work(2, *context, kernel);
        gpu::Workspace device_work(2, *context, kernel, gpu::ExecutionMode::device);
        for (const bool fused : {false, true})
        {
            SCOPED_TRACE(fused);
            for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
            {
                SCOPED_TRACE(static_cast<int>(algorithm));
                // Reuse the same resources after breakdown, a restart, and
                // iteration-limit exits, then repeat the original breakdown.
                for (const auto limits : {std::array<std::size_t, 2>{3, 0}, {3, 1}, {1, 0}, {0, 0}, {3, 0}})
                {
                    const auto maxiter = limits[0];
                    const auto refresh = limits[1];
                    SCOPED_TRACE(maxiter);
                    SCOPED_TRACE(refresh);
                    cg::CgOptions options;
                    options.algorithm = algorithm;
                    options.rtol = 0.0;
                    options.atol = 0.0;
                    options.maxiter = maxiter;
                    options.residual_refresh = refresh;
                    x_host.fill(0.0, *context);
                    x_device.fill(0.0, *context);
                    const auto reference = gpu::solve_into(A, b.view(), x_host.view(), host_work, options, fused);
                    const auto result = gpu::solve_into(A, b.view(), x_device.view(), device_work, options, fused);
                    const bool restarted = maxiter > 1 && refresh == 1 && algorithm == cg::Algorithm::recurrence;
                    const auto expected_status =
                        maxiter > 1 ? cg::Status::non_positive_curvature : cg::Status::max_iterations;
                    const std::size_t expected_matvecs = maxiter == 0 ? 1 : maxiter == 1 ? 3 : restarted ? 5 : 4;
                    EXPECT_EQ(result.status, expected_status);
                    EXPECT_EQ(result.iterations, maxiter == 0 ? 0u : 1u);
                    EXPECT_EQ(result.matvecs, expected_matvecs);
                    EXPECT_EQ(result.restarts, restarted ? 1u : 0u);
                    EXPECT_EQ(result.status, reference.status);
                    EXPECT_EQ(result.iterations, reference.iterations);
                    EXPECT_EQ(result.matvecs, reference.matvecs);
                    EXPECT_EQ(result.restarts, reference.restarts);
                    EXPECT_NEAR(result.residual_norm, reference.residual_norm, 1e-12);
                    std::vector<double> expected(2), actual(2);
                    x_host.download(expected, *context);
                    x_device.download(actual, *context);
                    for (std::size_t i = 0; i < actual.size(); ++i)
                    {
                        EXPECT_NEAR(actual[i], expected[i], 1e-12);
                        const double first_step = maxiter == 0 ? 0.0 : (i == 0 ? 10.0 / 3.0 : 5.0 / 3.0);
                        EXPECT_NEAR(actual[i], first_step, 1e-12);
                    }
                    EXPECT_NEAR(result.residual_norm, true_norm(host.view(), rhs, actual), 1e-12);
                }
            }
        }
    };
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        SCOPED_TRACE(static_cast<int>(kernel));
        const gpu::DenseMatrix A(host.view(), *context);
        check(A.view(), kernel);
    }
    const gpu::CsrMatrix sparse(as_csr(host.view()), *context);
    check(sparse, gpu::DenseKernel::native);
}

TEST_F(Cuda, DeviceResidualExitPrecedesNextRhoFailureRefreshAndIterationLimit)
{
    for (const std::size_t size : {2u, 257u})
    {
        SCOPED_TRACE(size);
        std::vector<double> rhs(size, 0.0);
        rhs.front() = 1.0;
        gpu::Vector b(rhs, *context), x(size, *context);
        for (const double off_diagonal : {0.0, 1e200})
        {
            SCOPED_TRACE(off_diagonal);
            // Both systems take the exact first step x=e0. Identity converges
            // with next-rho=0; finite off-diagonal residuals overflow their squared
            // sum. Size257 exercises separate lanes and a second lane-zero item.
            // Residual policy must win before beta/last/refresh.
            cg::DenseMatrix host(size, size);
            for (std::size_t row = 0; row < size; ++row)
            {
                host.data()[row * size + row] = 1.0;
            }
            host.data()[size] = off_diagonal;
            if (size > 2)
            {
                host.data()[(size - 1) * size] = off_diagonal;
            }
            const double expected_norm = size == 2 ? off_diagonal : std::hypot(off_diagonal, off_diagonal);
            const auto check = [&](const auto &A, gpu::DenseKernel kernel)
            {
                const gpu::JacobiPreconditioner M(A, *context);
                gpu::Workspace workspace(size, *context, kernel, gpu::ExecutionMode::device);
                for (const bool fused : {false, true})
                {
                    SCOPED_TRACE(fused);
                    for (const auto algorithm : {cg::Algorithm::recurrence, cg::Algorithm::explicit_residual})
                    {
                        SCOPED_TRACE(static_cast<int>(algorithm));
                        for (const bool preconditioned : {false, true})
                        {
                            if (preconditioned && algorithm != cg::Algorithm::recurrence)
                            {
                                continue;
                            }
                            SCOPED_TRACE(preconditioned);
                            for (const std::size_t maxiter : {1u, 2u})
                            {
                                SCOPED_TRACE(maxiter);
                                cg::CgOptions options;
                                options.algorithm = algorithm;
                                options.rtol = 0.0;
                                options.atol = 0.0;
                                options.maxiter = maxiter;
                                options.residual_refresh = 1;
                                x.fill(0.0, *context);
                                const auto result = gpu::solve_into(A, b.view(), x.view(), workspace, options, fused,
                                                                    preconditioned ? &M : nullptr);
                                EXPECT_EQ(result.status,
                                          off_diagonal == 0.0 ? cg::Status::converged : cg::Status::non_finite);
                                EXPECT_EQ(result.iterations, 1u);
                                EXPECT_EQ(result.matvecs, 3u);
                                EXPECT_EQ(result.restarts, 0u);
                                EXPECT_DOUBLE_EQ(result.residual_norm, expected_norm);
                                std::vector<double> actual(size), expected(size, 0.0);
                                expected.front() = 1.0;
                                x.download(actual, *context);
                                EXPECT_EQ(actual, expected);
                                EXPECT_DOUBLE_EQ(result.residual_norm, true_norm(host.view(), rhs, actual));
                            }
                        }
                    }
                }
            };
            for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
            {
                SCOPED_TRACE(static_cast<int>(kernel));
                const gpu::DenseMatrix A(host.view(), *context);
                check(A.view(), kernel);
            }
            const gpu::CsrMatrix sparse(as_csr(host.view()), *context);
            check(sparse, gpu::DenseKernel::native);
        }
    }
}

TEST_F(Cuda, AsyncMetricsKeepIndependentDeviceResultsAndStableNormFallbacks)
{
    gpu::detail::ExecutionResources resources(*context);
    const gpu::detail::Backend backend(resources);
    const auto metrics = gpu::detail::device_allocation<double>(4, *context);
    for (const std::size_t size : {0u, 1u, 255u, 256u, 257u, 513u})
    {
        SCOPED_TRACE(size);
        const std::vector<double> rhs(size, 2.0);
        gpu::Vector y(rhs, *context), x(size, *context);
        for (const double magnitude : {0.0, 3.0, 1e-200, 1e-160, 1e200, std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::quiet_NaN()})
        {
            SCOPED_TRACE(magnitude);
            std::vector<double> values(size, 0.0);
            if (size != 0)
            {
                values.front() = magnitude;
            }
            if (size > 1)
            {
                values[1] = -magnitude;
            }
            if (size > 2)
            {
                values.back() = magnitude;
            }
            double expected_norm = 0.0;
            for (const double value : values)
            {
                expected_norm = std::hypot(expected_norm, value);
            }
            x.upload(values, *context);
            backend.dot_into(x.view(), x.view(), metrics.get());
            backend.norm_from_squared_into(x.view(), metrics.get(), metrics.get() + 1);
            backend.dot_into(y.view(), y.view(), metrics.get() + 2);
            backend.norm_from_squared_into(y.view(), metrics.get() + 2, metrics.get() + 3);
            std::array<double, 4> result{};
            ASSERT_EQ(cudaMemcpyAsync(result.data(), metrics.get(), sizeof(result), cudaMemcpyDeviceToHost,
                                      reinterpret_cast<cudaStream_t>(context->stream())),
                      cudaSuccess);
            context->synchronize();
            if (std::isnan(expected_norm))
            {
                EXPECT_TRUE(std::isnan(result[1]));
            }
            else
            {
                EXPECT_DOUBLE_EQ(result[1], expected_norm);
            }
            EXPECT_DOUBLE_EQ(result[2], 4.0 * size);
            EXPECT_DOUBLE_EQ(result[3], std::sqrt(4.0 * size));
        }
    }
}

TEST_F(Cuda, CsrMovesTransferStorageAndLeaveReusableEmptySources)
{
    gpu::CsrMatrix source(cg::CsrMatrix(2, 2, {0, 1, 2}, {0, 1}, {2, 3}), *context);
    const auto original = source.view();
    const auto check_empty = [](const gpu::CsrMatrix &A)
    {
        EXPECT_EQ(A.rows(), 0u);
        EXPECT_EQ(A.cols(), 0u);
        EXPECT_EQ(A.nnz(), 0u);
        EXPECT_EQ(A.device(), -1);
        EXPECT_EQ(A.view().indptr(), nullptr);
        EXPECT_EQ(A.view().indices(), nullptr);
        EXPECT_EQ(A.view().data(), nullptr);
        EXPECT_EQ(A.view().rows(), 0u);
        EXPECT_EQ(A.view().cols(), 0u);
        EXPECT_EQ(A.view().nnz(), 0u);
        EXPECT_EQ(A.view().device(), -1);
    };
    const auto check_storage = [&](const gpu::CsrMatrix &A)
    {
        EXPECT_EQ(A.rows(), 2u);
        EXPECT_EQ(A.cols(), 2u);
        EXPECT_EQ(A.nnz(), 2u);
        EXPECT_EQ(A.device(), context->device());
        EXPECT_EQ(A.view().indptr(), original.indptr());
        EXPECT_EQ(A.view().indices(), original.indices());
        EXPECT_EQ(A.view().data(), original.data());
        EXPECT_EQ(A.view().index_type(), gpu::IndexType::int64);
    };
    gpu::CsrMatrix moved(std::move(source));
    check_empty(source);
    check_storage(moved);
    auto &same = moved;
    moved = std::move(same);
    check_storage(moved);

    gpu::CsrMatrix destination(cg::CsrMatrix(1, 1, {0, 1}, {0}, {7}), *context);
    destination = std::move(moved);
    check_empty(moved);
    check_storage(destination);
    source = std::move(destination);
    check_empty(destination);
    check_storage(source);

    gpu::Vector b(std::vector<double>{1, 1}, *context), x(2, *context);
    gpu::Workspace workspace(2, *context);
    x.fill(0, *context);
    ASSERT_EQ(gpu::solve_into(source, b.view(), x.view(), workspace).status, cg::Status::converged);
    std::vector<double> result(2);
    x.download(result, *context);
    EXPECT_NEAR(result[0], 0.5, 1e-14);
    EXPECT_NEAR(result[1], 1.0 / 3.0, 1e-14);
}

TEST_F(Cuda, EmptyCsrSnapshotsRetainTheCanonicalRowPointerAndDevice)
{
    const gpu::CsrMatrix host_snapshot(cg::CsrMatrix(0, 0, {0}, {}, {}), *context);
    const gpu::CsrMatrix device_snapshot(host_snapshot.view(), *context);
    EXPECT_NE(device_snapshot.view().indptr(), host_snapshot.view().indptr());
    for (const auto *A : {&host_snapshot, &device_snapshot})
    {
        EXPECT_EQ(A->rows(), 0u);
        EXPECT_EQ(A->cols(), 0u);
        EXPECT_EQ(A->nnz(), 0u);
        EXPECT_EQ(A->device(), context->device());
        EXPECT_EQ(A->view().device(), context->device());
        EXPECT_EQ(A->view().index_type(), gpu::IndexType::int64);
        ASSERT_NE(A->view().indptr(), nullptr);
        EXPECT_EQ(A->view().indices(), nullptr);
        EXPECT_EQ(A->view().data(), nullptr);
        std::int64_t row_pointer = -1;
        ASSERT_EQ(cudaMemcpy(&row_pointer, A->view().indptr(), sizeof(row_pointer), cudaMemcpyDeviceToHost),
                  cudaSuccess);
        EXPECT_EQ(row_pointer, 0);
    }
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
            gpu::PreparedSolver plan(gpu::DenseMatrix(host.view(), *context), *context, kind, kernel);
            const auto moved = std::move(plan);
            check(moved, kind);
        }
        const gpu::PreparedSolver sparse(gpu::CsrMatrix(as_csr(host.view()), *context), *context, kind,
                                         gpu::DenseKernel::native);
        check(sparse, kind);
    }
    EXPECT_THROW(gpu::PreparedSolver(gpu::CsrMatrix(as_csr(host.view()), *context), *context,
                                     cg::PreconditionerKind::none, gpu::DenseKernel::cublas),
                 std::invalid_argument);
    const auto invalid_kind = static_cast<cg::PreconditionerKind>(-1);
    EXPECT_THROW(
        gpu::PreparedSolver(gpu::DenseMatrix(host.view(), *context), *context, invalid_kind, gpu::DenseKernel::native),
        std::invalid_argument);
    EXPECT_THROW(gpu::PreparedSolver(gpu::CsrMatrix(as_csr(host.view()), *context), *context, invalid_kind,
                                     gpu::DenseKernel::native),
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
        static_cast<void>(
            gpu::solve_into(A.view(), b.view(), x.view(), workspace, options, true, cg::PreconditionerKind::jacobi));
        FAIL() << "explicit CG must reject Jacobi before reading its diagonal";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_STREQ(error.what(), "Jacobi preconditioning requires recurrence CG");
    }
    EXPECT_THROW(static_cast<void>(gpu::solve_into(A.view(), b.view(), x.view(), workspace, {}, true,
                                                   static_cast<cg::PreconditionerKind>(-1))),
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
        EXPECT_NE(A.view().indptr(), view.indptr());
        EXPECT_NE(A.view().indices(), view.indices());
        EXPECT_NE(A.view().data(), view.data());
        data.fill(99, *context);
        x.fill(0, *context);
        gpu::Workspace workspace(2, *context);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        ASSERT_EQ(gpu::solve_into(A, b.view(), x.view(), workspace, options).status, cg::Status::converged);
        std::vector<double> result(2);
        x.download(result, *context);
        EXPECT_NEAR(result[0], 0.5, 1e-14);
        EXPECT_NEAR(result[1], 1.0 / 3.0, 1e-14);
        data.upload(values, *context);
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

TEST_F(Cuda, JacobiOddInverseSnapshotSurvivesMoveConstructionAssignmentAndSelfMove)
{
    const cg::DenseMatrix host(3, 3, {3, 0, 0, 0, 7, 0, 0, 0, 11});
    const std::vector<double> rhs{1, 2, 3};
    gpu::Vector r(rhs, *context), z(3, *context);
    gpu::detail::ExecutionResources resources(*context);
    const gpu::detail::Backend backend(resources);
    const auto empty = [](const gpu::JacobiPreconditioner &M)
    {
        EXPECT_EQ(M.size(), 0u);
        EXPECT_EQ(M.device(), -1);
        EXPECT_EQ(M.inverse_diagonal().data(), nullptr);
        EXPECT_EQ(M.inverse_diagonal().size(), 0u);
    };
    const auto check = [&](gpu::JacobiPreconditioner source)
    {
        const auto *allocation = source.inverse_diagonal().data();
        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(allocation) % alignof(double), 0u);
        gpu::JacobiPreconditioner moved(std::move(source));
        empty(source);
        auto &same = moved;
        moved = std::move(same);
        EXPECT_EQ(moved.inverse_diagonal().data(), allocation);
        const gpu::DenseMatrix replacement(host.view(), *context);
        gpu::JacobiPreconditioner assigned(replacement.view(), *context);
        assigned = std::move(moved);
        empty(moved);
        source = std::move(assigned);
        empty(assigned);
        EXPECT_EQ(source.size(), 3u);
        EXPECT_EQ(source.device(), context->device());
        EXPECT_EQ(source.inverse_diagonal().data(), allocation);
        EXPECT_EQ(source.inverse_diagonal().size(), 3u);
        std::vector<double> inverse(3), output(3);
        ASSERT_EQ(
            cudaMemcpy(inverse.data(), source.inverse_diagonal().data(), 3 * sizeof(double), cudaMemcpyDeviceToHost),
            cudaSuccess);
        EXPECT_DOUBLE_EQ(inverse[0], 1.0 / 3.0);
        EXPECT_DOUBLE_EQ(inverse[1], 1.0 / 7.0);
        EXPECT_DOUBLE_EQ(inverse[2], 1.0 / 11.0);
        EXPECT_NEAR(backend.multiply_and_dot(source.inverse_diagonal(), r.view(), z.view()),
                    1.0 / 3.0 + 4.0 / 7.0 + 9.0 / 11.0, 1e-14);
        z.download(output, *context);
        for (std::size_t i = 0; i < output.size(); ++i)
        {
            EXPECT_NEAR(output[i], rhs[i] * inverse[i], 1e-14);
        }
    };
    // Both source matrices are destroyed before the preconditioner is applied.
    check(
        [&]
        {
            const gpu::DenseMatrix A(host.view(), *context);
            return gpu::JacobiPreconditioner(A.view(), *context);
        }());
    check(
        [&]
        {
            const gpu::CsrMatrix A(as_csr(host.view()), *context);
            return gpu::JacobiPreconditioner(A, *context);
        }());
}

TEST_F(Cuda, JacobiRejectsInverseStorageOverflowBeforeAllocation)
{
    const auto size = std::numeric_limits<std::size_t>::max() / sizeof(double) + 1;
    const gpu::DenseMatrixView too_large(nullptr, size, size, size, context->device());
    try
    {
        gpu::JacobiPreconditioner M(too_large, *context);
        FAIL() << "overflowing inverse storage must be rejected";
    }
    catch (const std::invalid_argument &error)
    {
        EXPECT_STREQ(error.what(), "storage size overflows size_t");
    }
}

TEST_F(Cuda, OneShotJacobiValidationFailurePreservesOutputAndAllowsFreshSetupOnReuse)
{
    const cg::DenseMatrix valid(3, 3, {3, 0, 0, 0, 7, 0, 0, 0, 11});
    const cg::DenseMatrix invalid(3, 3, {3, 0, 0, 0, 7, 0, 0, 0, -11});
    const gpu::DenseMatrix dense_good(valid.view(), *context), dense_bad(invalid.view(), *context);
    const gpu::CsrMatrix csr_good(as_csr(valid.view()), *context), csr_bad(as_csr(invalid.view()), *context);
    const std::vector<double> rhs{1, 2, 3};
    gpu::Vector b(rhs, *context), x(3, *context);
    for (const auto mode : {gpu::ExecutionMode::host, gpu::ExecutionMode::device})
    {
        gpu::Workspace workspace(3, *context, gpu::DenseKernel::native, mode);
        const auto check = [&](const auto &bad, const auto &good)
        {
            x.fill(0, *context);
            EXPECT_THROW(static_cast<void>(gpu::solve_into(bad, b.view(), x.view(), workspace, {}, true,
                                                           cg::PreconditionerKind::jacobi)),
                         std::invalid_argument);
            std::vector<double> output(3);
            x.download(output, *context);
            EXPECT_EQ(output, (std::vector<double>{0, 0, 0}));
            const auto result =
                gpu::solve_into(good, b.view(), x.view(), workspace, {}, true, cg::PreconditionerKind::jacobi);
            EXPECT_EQ(result.status, cg::Status::converged);
            EXPECT_EQ(result.iterations, 1u);
        };
        check(dense_bad.view(), dense_good.view());
        check(csr_bad, csr_good);
    }
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
                const auto info = gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, options, true);
                EXPECT_EQ(info.status, cg::Status::non_finite);
                EXPECT_EQ(info.iterations, 0u);
                EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
                options.maxiter = 0;
                EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, options, true).status,
                          cg::Status::max_iterations);
                if (magnitude == 1e-200)
                {
                    options.atol = 1e-199;
                    EXPECT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), selected_workspace, options, true).status,
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
            const cg::DenseMatrix host_negative(2, 2, {-1, 0, 0, -1});
            const gpu::DenseMatrix negative(host_negative.view(), *context);
            x.fill(0, *context);
            EXPECT_EQ(gpu::solve_into(negative.view(), b.view(), x.view(), selected_workspace, options, true).status,
                      cg::Status::non_positive_curvature);
            const cg::DenseMatrix host_invalid(2, 2, {1, INFINITY, 0, 1});
            const gpu::DenseMatrix invalid(host_invalid.view(), *context);
            x.fill(0, *context);
            EXPECT_EQ(gpu::solve_into(invalid.view(), b.view(), x.view(), selected_workspace, options, true).status,
                      cg::Status::non_finite);
        }
    }
}

TEST_F(Cuda, NondefaultStreamIsBorrowedAndSelectedByWorkspace)
{
    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess);
    {
        const gpu::Context selected(0, reinterpret_cast<std::uintptr_t>(stream));
        const auto host = system_matrix(17);
        const gpu::DenseMatrix A(host.view(), selected);
        const std::vector<double> hb(17, 1);
        gpu::Vector b(hb, selected), x(17, selected);
        cg::CgOptions options;
        options.algorithm = cg::Algorithm::recurrence;
        for (const auto mode : {gpu::ExecutionMode::host, gpu::ExecutionMode::device})
        {
            x.fill(0, selected);
            gpu::Workspace workspace(17, selected, gpu::DenseKernel::native, mode);
            EXPECT_EQ(workspace.context().device(), selected.device());
            EXPECT_EQ(workspace.context().stream(), reinterpret_cast<std::uintptr_t>(stream));
            ASSERT_EQ(gpu::solve_into(A.view(), b.view(), x.view(), workspace, options).status, cg::Status::converged);
            const auto repeated = gpu::solve_into(A.view(), b.view(), x.view(), workspace, options);
            EXPECT_EQ(repeated.status, cg::Status::converged);
            EXPECT_EQ(repeated.iterations, 0u);
            EXPECT_EQ(repeated.matvecs, 1u);
        }
        unsigned int flags = 0;
        EXPECT_EQ(cudaStreamGetFlags(stream, &flags), cudaSuccess);
    }
    EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
}

TEST_F(Cuda, ImmutablePreparedPlanSupportsConcurrentSolvesOnIndependentStreams)
{
    const auto host = system_matrix(33);
    for (const auto mode : {gpu::ExecutionMode::host, gpu::ExecutionMode::device})
    {
        for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
        {
            const gpu::PreparedSolver plan(gpu::DenseMatrix(host.view(), *context), *context,
                                           cg::PreconditionerKind::jacobi, kernel);
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
                    auto workspace = plan.create_workspace(stream_address, mode);
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
}

TEST_F(Cuda, BlockDotHandlesWarpBlockAndGridStrideTails)
{
    gpu::detail::ExecutionResources resources(*context);
    const gpu::detail::Backend backend(resources);
    const std::size_t sizes[] = {0, 1, 31, 32, 33, 255, 256, 257, 65537};
    for (const auto size : sizes)
    {
        SCOPED_TRACE(size);
        std::vector<double> host_x(size), host_y(size);
        long double expected_dot = 0.0L, expected_squared = 0.0L;
        for (std::size_t i = 0; i < size; ++i)
        {
            host_x[i] = static_cast<double>(static_cast<int>(i % 17) - 8);
            host_y[i] = static_cast<double>(static_cast<int>((i * 3) % 13) - 6);
            expected_dot += static_cast<long double>(host_x[i]) * host_y[i];
            expected_squared += static_cast<long double>(host_x[i]) * host_x[i];
        }
        gpu::Vector x(host_x, *context), y(host_y, *context);
        // Every product and partial sum is an exactly representable integer.
        EXPECT_DOUBLE_EQ(backend.dot(x.view(), y.view()), static_cast<double>(expected_dot));
        EXPECT_DOUBLE_EQ(backend.dot(x.view(), x.view()), static_cast<double>(expected_squared));
        EXPECT_DOUBLE_EQ(backend.norm(x.view()), std::sqrt(static_cast<double>(expected_squared)));
    }
}

TEST_F(Cuda, BlockReductionPreservesNonfiniteValuesOutsideTheLeadingLane)
{
    constexpr std::size_t size = 257;
    std::vector<double> host_x(size, 1.0), host_y(size, 1.0);
    gpu::Vector x(host_x, *context), y(host_y, *context);
    gpu::detail::ExecutionResources resources(*context);
    const gpu::detail::Backend backend(resources);
    const std::size_t positions[] = {1, 31, 255, 256};
    for (const auto position : positions)
    {
        SCOPED_TRACE(position);
        std::fill(host_x.begin(), host_x.end(), 1.0);
        host_x[position] = std::numeric_limits<double>::quiet_NaN();
        x.upload(host_x, *context);
        EXPECT_TRUE(std::isnan(backend.dot(x.view(), y.view())));
        EXPECT_TRUE(std::isnan(backend.norm(x.view())));

        host_x[position] = std::numeric_limits<double>::infinity();
        x.upload(host_x, *context);
        EXPECT_EQ(backend.dot(x.view(), y.view()), std::numeric_limits<double>::infinity());
        EXPECT_EQ(backend.norm(x.view()), std::numeric_limits<double>::infinity());

        host_x[position] = -std::numeric_limits<double>::infinity();
        x.upload(host_x, *context);
        EXPECT_EQ(backend.dot(x.view(), y.view()), -std::numeric_limits<double>::infinity());
        EXPECT_EQ(backend.norm(x.view()), std::numeric_limits<double>::infinity());
    }
    std::fill(host_x.begin(), host_x.end(), 1.0);
    host_x[1] = std::numeric_limits<double>::infinity();
    host_x[255] = -std::numeric_limits<double>::infinity();
    x.upload(host_x, *context);
    EXPECT_TRUE(std::isnan(backend.dot(x.view(), y.view())));
    EXPECT_EQ(backend.norm(x.view()), std::numeric_limits<double>::infinity());
}

TEST_F(Cuda, BorrowedPaddedDeviceMatrixReusesBlockStorageAcrossGridStrideRows)
{
    constexpr std::size_t rows = 513, cols = 257, row_stride = 264;
    std::vector<double> host_matrix(rows * row_stride, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> host_x(cols), host_b(rows), expected_product(rows), downloaded(rows);
    long double expected_squared = 0.0L;
    for (std::size_t column = 0; column < cols; ++column)
    {
        host_x[column] = static_cast<double>(static_cast<int>(column % 7) - 3);
    }
    for (std::size_t row = 0; row < rows; ++row)
    {
        host_b[row] = static_cast<double>(static_cast<int>(row % 11) - 5);
        long double product = 0.0L;
        for (std::size_t column = 0; column < cols; ++column)
        {
            const double value = static_cast<double>(static_cast<int>((row * 3 + column * 5) % 17) - 8);
            host_matrix[row * row_stride + column] = value;
            product += static_cast<long double>(value) * host_x[column];
        }
        expected_product[row] = static_cast<double>(product);
        const long double residual = static_cast<long double>(host_b[row]) - product;
        expected_squared += residual * residual;
    }
    gpu::Vector matrix_storage(host_matrix, *context), x(host_x, *context), b(host_b, *context), output(rows, *context);
    // An owning DenseMatrix upload would pack away the padding. Borrow the
    // device buffer directly so every kernel actually sees row_stride=264.
    const gpu::DenseMatrixView A(matrix_storage.data(), rows, cols, row_stride, context->device());
    for (const auto kernel : {gpu::DenseKernel::native, gpu::DenseKernel::cublas})
    {
        SCOPED_TRACE(kernel == gpu::DenseKernel::native ? "native" : "cublas");
        gpu::detail::ExecutionResources resources(*context, kernel);
        for (const bool fused : {false, true})
        {
            SCOPED_TRACE(fused);
            const gpu::detail::Backend backend(resources, fused);
            output.fill(std::numeric_limits<double>::quiet_NaN(), *context);
            backend.apply(A, x.view(), output.view());
            output.download(downloaded, *context);
            for (std::size_t row = 0; row < rows; ++row)
            {
                EXPECT_DOUBLE_EQ(downloaded[row], expected_product[row]) << "row=" << row;
            }
            for (const auto sign : {cg::ResidualSign::rhs_minus_product, cg::ResidualSign::product_minus_rhs})
            {
                EXPECT_DOUBLE_EQ(backend.residual_squared_norm(A, b.view(), x.view(), output.view(), sign),
                                 static_cast<double>(expected_squared));
                output.download(downloaded, *context);
                for (std::size_t row = 0; row < rows; ++row)
                {
                    const double residual = host_b[row] - expected_product[row];
                    EXPECT_DOUBLE_EQ(downloaded[row],
                                     sign == cg::ResidualSign::rhs_minus_product ? residual : -residual)
                        << "row=" << row;
                }
            }
        }
    }
}

} // namespace
