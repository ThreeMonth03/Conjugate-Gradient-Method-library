#include <cg/cg.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace
{

cg::DenseMatrix scaled_tridiagonal(std::size_t size)
{
    cg::DenseMatrix A(size, size);
    std::vector<double> scales(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        scales[i] = std::pow(100.0, static_cast<double>(i) / static_cast<double>(size - 1));
        A.data()[i * size + i] = 2.0 * scales[i] * scales[i];
        if (i != 0)
        {
            const double off_diagonal = -scales[i] * scales[i - 1];
            A.data()[i * size + i - 1] = off_diagonal;
            A.data()[(i - 1) * size + i] = off_diagonal;
        }
    }
    return A;
}

cg::CsrMatrix as_csr(cg::DenseMatrixView A)
{
    std::vector<std::size_t> indptr{0}, indices;
    std::vector<double> values;
    for (std::size_t row = 0; row < A.rows; ++row)
    {
        for (std::size_t column = 0; column < A.cols; ++column)
        {
            const auto value = A.data[row * A.row_stride + column];
            if (value != 0)
            {
                indices.push_back(column);
                values.push_back(value);
            }
        }
        indptr.push_back(indices.size());
    }
    return {A.rows, A.cols, std::move(indptr), std::move(indices), std::move(values)};
}

TEST(Jacobi, OwnsItsInverseDiagonalSnapshot)
{
    cg::DenseMatrix A(3, 3, {2, 0, 0, 0, 4, 0, 0, 0, 8});
    const cg::JacobiPreconditioner M(A.view());
    A.data()[0] = 100;
    EXPECT_EQ(M.size(), 3u);
    EXPECT_DOUBLE_EQ(M.inverse_diagonal().data()[0], 0.5);
    EXPECT_DOUBLE_EQ(M.inverse_diagonal().data()[1], 0.25);
    EXPECT_DOUBLE_EQ(M.inverse_diagonal().data()[2], 0.125);
    std::vector<double> r({4, 8, 16}), z(3);
    const cg::CpuBackend backend;
    M.apply(r, z, backend);
    for (std::size_t i = 0; i < z.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(z.data()[i], 2);
    }
    EXPECT_DOUBLE_EQ(M.apply_and_dot(r, z, backend), 56);
}

TEST(Jacobi, RejectsMissingNonpositiveNonfiniteAndUnrepresentableDiagonal)
{
    for (const double invalid : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                                 std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::denorm_min()})
    {
        const cg::DenseMatrix dense(2, 2, {1, 0, 0, invalid});
        const cg::CsrMatrix sparse(2, 2, {0, 1, 2}, {0, 1}, {1, invalid});
        EXPECT_THROW(cg::JacobiPreconditioner{dense.view()}, std::invalid_argument);
        EXPECT_THROW(cg::JacobiPreconditioner{sparse}, std::invalid_argument);
    }
    const cg::CsrMatrix missing(2, 2, {0, 1, 1}, {0}, {1});
    EXPECT_THROW(cg::JacobiPreconditioner{missing}, std::invalid_argument);
    const cg::DenseMatrix rectangular(2, 3);
    EXPECT_THROW(cg::JacobiPreconditioner{rectangular.view()}, std::invalid_argument);
    const cg::DenseMatrix empty(0, 0);
    EXPECT_THROW(cg::JacobiPreconditioner{empty.view()}, std::invalid_argument);
}

TEST(Jacobi, FusedApplyMatchesReferenceIncludingParallelVectorTails)
{
    const std::size_t size = 16385;
    std::vector<std::size_t> indptr(size + 1), indices(size);
    std::vector<double> values(size);
    std::vector<double> r(size), z(size), reference_z(size);
    for (std::size_t i = 0; i < size; ++i)
    {
        indptr[i] = indices[i] = i;
        values[i] = 1.0 + static_cast<double>(i % 37);
        r.data()[i] = std::sin(static_cast<double>(i));
    }
    indptr[size] = size;
    const cg::CsrMatrix A(size, size, std::move(indptr), std::move(indices), std::move(values));
    const cg::JacobiPreconditioner M(A);
    const cg::CpuBackend reference(1, false);
    const int threads = cg::CpuBackend::openmp_enabled() ? 2 : 1;
    const cg::CpuBackend fused(threads, true);
    const double expected = M.apply_and_dot(r, reference_z, reference);
    EXPECT_NEAR(M.apply_and_dot(r, z, fused), expected, expected * 1e-12);
    for (std::size_t i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(z.data()[i], reference_z.data()[i]);
    }
}

TEST(PCG, JacobiReducesIterationsOnADiagonallyScaledSPDSystem)
{
    const auto A = scaled_tridiagonal(32);
    const cg::JacobiPreconditioner M(A.view());
    std::vector<double> truth(32), b(32), x(32), plain_x(32);
    for (std::size_t i = 0; i < truth.size(); ++i)
    {
        truth.data()[i] = 0.5 + std::sin(static_cast<double>(i));
    }
    const cg::CpuBackend backend;
    backend.apply(A.view(), truth, b);
    cg::Workspace workspace(32), plain_workspace(32);
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.residual_refresh = 0;
    options.rtol = 1e-10;
    options.maxiter = 2000;
    const auto plain = cg::solve_into(A.view(), b, plain_x, plain_workspace, backend, options);
    const auto pcg = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(plain.status, cg::Status::converged);
    EXPECT_EQ(pcg.status, cg::Status::converged);
    EXPECT_LT(pcg.iterations, plain.iterations);
    EXPECT_LE(pcg.residual_norm, options.rtol * std::sqrt(backend.squared_norm(b)));
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        EXPECT_NEAR(x.data()[i], truth.data()[i], 1e-6);
    }
}

TEST(PCG, DenseAndCSRSolversUseTheSamePreconditionedRecurrence)
{
    const auto dense = scaled_tridiagonal(17);
    const auto sparse = as_csr(dense.view());
    const cg::JacobiPreconditioner dense_M(dense.view()), sparse_M(sparse);
    std::vector<double> b(17), dense_x(17), sparse_x(17), check(17);
    for (std::size_t i = 0; i < b.size(); ++i)
    {
        b.data()[i] = 0.5 + std::cos(static_cast<double>(i));
        EXPECT_DOUBLE_EQ(dense_M.inverse_diagonal().data()[i], sparse_M.inverse_diagonal().data()[i]);
    }
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.rtol = 1e-10;
    options.residual_refresh = 0;
    for (const bool fused : {false, true})
    {
        std::fill_n(dense_x.data(), dense_x.size(), 0.0);
        std::fill_n(sparse_x.data(), sparse_x.size(), 0.0);
        const cg::CpuBackend backend(1, fused);
        cg::Workspace dense_workspace(17), sparse_workspace(17);
        const auto dense_info = cg::solve_into(dense.view(), b, dense_x, dense_workspace, backend, options, &dense_M);
        const auto sparse_info = cg::solve_into(sparse, b, sparse_x, sparse_workspace, backend, options, &sparse_M);
        EXPECT_EQ(dense_info.status, cg::Status::converged);
        EXPECT_EQ(sparse_info.status, cg::Status::converged);
        EXPECT_DOUBLE_EQ(sparse_info.residual_norm,
                         std::sqrt(backend.residual_squared_norm(sparse, b, sparse_x, check)));
        for (std::size_t i = 0; i < b.size(); ++i)
        {
            EXPECT_NEAR(dense_x.data()[i], sparse_x.data()[i], 1e-7);
        }
    }
}

TEST(PCG, RefreshRestartsWithPreconditionedTrueResidualAndWorkspaceCanBeReused)
{
    const cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    const cg::JacobiPreconditioner M(A.view());
    std::vector<double> b({1, 2}), x(2), check(2);
    cg::Workspace workspace(2);
    const auto *z_pointer = workspace.preconditioned().data();
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    options.residual_refresh = 1;
    options.rtol = 1e-10;
    for (std::size_t call = 0; call < 2; ++call)
    {
        std::fill_n(x.data(), x.size(), 0.0);
        const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
        EXPECT_EQ(info.status, cg::Status::converged);
        EXPECT_GT(info.restarts, 0u);
        EXPECT_EQ(info.restarts, info.iterations - 1);
        EXPECT_EQ(info.matvecs, 1 + 2 * info.iterations);
        EXPECT_EQ(workspace.preconditioned().data(), z_pointer);
        EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(backend.residual_squared_norm(A.view(), b, x, check)));
        EXPECT_NEAR(x.data()[0], 1.0 / 11, 1e-9);
        EXPECT_NEAR(x.data()[1], 7.0 / 11, 1e-9);
    }
}

TEST(PCG, RejectsIncompatibleAlgorithmDimensionsAndOutputAliasingBeforeMutation)
{
    const cg::DenseMatrix A(2, 2, {4, 1, 1, 3}), small(1, 1, {1});
    const cg::JacobiPreconditioner M(A.view()), wrong_size(small.view());
    std::vector<double> b({1, 2}), x({10, 20});
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::explicit_residual;
    EXPECT_THROW(cg::solve_into(A.view(), b, x, workspace, backend, options, &M), std::invalid_argument);
    EXPECT_DOUBLE_EQ(x.data()[0], 10);
    EXPECT_DOUBLE_EQ(x.data()[1], 20);
    options.algorithm = cg::Algorithm::recurrence;
    EXPECT_THROW(cg::solve_into(A.view(), b, x, workspace, backend, options, &wrong_size), std::invalid_argument);
    cg::VectorView inverse_output{const_cast<double *>(M.inverse_diagonal().data()), M.size()};
    EXPECT_THROW(cg::solve_into(A.view(), b, inverse_output, workspace, backend, options, &M), std::invalid_argument);
    EXPECT_THROW(M.apply(b, inverse_output, backend), std::invalid_argument);
    EXPECT_THROW(cg::solve_into(A.view(), b, workspace.preconditioned(), workspace, backend, options, &M),
                 std::invalid_argument);
    EXPECT_THROW(M.apply_and_dot(b, b, backend), std::invalid_argument);
    EXPECT_DOUBLE_EQ(M.inverse_diagonal().data()[0], 0.25);
}

TEST(PCG, InitialConvergenceAndZeroIterationsDoNotApplyThePreconditioner)
{
    const cg::DenseMatrix A(2, 2, {2, 0, 0, 3});
    const cg::JacobiPreconditioner M(A.view());
    std::vector<double> b({2, 3}), x({1, 1});
    cg::Workspace workspace(2);
    workspace.preconditioned().data()[0] = 123.0;
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    auto info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 1u);
    EXPECT_DOUBLE_EQ(workspace.preconditioned().data()[0], 123.0);
    std::fill_n(x.data(), x.size(), 0.0);
    options.maxiter = 0;
    info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::max_iterations);
    EXPECT_EQ(info.matvecs, 1u);
    EXPECT_DOUBLE_EQ(workspace.preconditioned().data()[0], 123.0);
}

TEST(PCG, ReportsPreconditionedOverflowWithTheFinalTrueResidual)
{
    const cg::DenseMatrix A(1, 1, {1e-200});
    const cg::JacobiPreconditioner M(A.view());
    std::vector<double> b(std::vector<double>{1e100}), x(1);
    cg::Workspace workspace(1);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::non_finite);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 2u);
    EXPECT_DOUBLE_EQ(info.residual_norm, 1e100);
    EXPECT_DOUBLE_EQ(x.data()[0], 0);
}

TEST(PCG, PositiveJacobiDiagonalDoesNotConcealAnIndefiniteMatrix)
{
    const cg::DenseMatrix A(2, 2, {1, 2, 2, 1});
    const cg::JacobiPreconditioner M(A.view());
    std::vector<double> b({1, -1}), x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::non_positive_curvature);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_EQ(info.matvecs, 3u);
    EXPECT_DOUBLE_EQ(info.residual_norm, std::sqrt(2.0));
}

TEST(PCGRange, IdentityAndJacobiDoNotHideSquaredResidualRangeFailure)
{
    const cg::DenseMatrix identity(2, 2, {1, 0, 0, 1});
    const cg::CsrMatrix sparse(2, 2, {0, 1, 2}, {0, 1}, {1, 1});
    const cg::JacobiPreconditioner M(identity.view());
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    for (const double magnitude : {1e-200, 1e-160, 1e200})
    {
        const std::vector<double> b({magnitude, -magnitude});
        std::vector<double> x(2);
        cg::Workspace workspace(2);
        auto info = cg::solve_into(identity.view(), b, x, workspace, backend, options, &M);
        EXPECT_EQ(info.status, cg::Status::non_finite);
        EXPECT_EQ(info.iterations, 0u);
        EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
        info = cg::solve_into(sparse, b, x, workspace, backend, options, &M);
        EXPECT_EQ(info.status, cg::Status::non_finite);
        EXPECT_NEAR(info.residual_norm, std::hypot(magnitude, magnitude), magnitude * 1e-14);
    }
    std::vector<double> zero(2), x(2);
    cg::Workspace workspace(2);
    const auto info = cg::solve_into(sparse, zero, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_DOUBLE_EQ(info.residual_norm, 0);
}

TEST(PCGRange, UnusablePreconditionedProductReportsAFiniteTrueNorm)
{
    const cg::DenseMatrix A(1, 1, {1e308});
    const cg::JacobiPreconditioner M(A.view());
    const std::vector<double> b(std::vector<double>{1.0});
    std::vector<double> x(1);
    cg::Workspace workspace(1);
    const cg::CpuBackend backend;
    cg::CgOptions options;
    options.algorithm = cg::Algorithm::recurrence;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, options, &M);
    EXPECT_EQ(info.status, cg::Status::non_finite);
    EXPECT_EQ(info.iterations, 0u);
    EXPECT_DOUBLE_EQ(info.residual_norm, 1.0);
}

} // namespace
