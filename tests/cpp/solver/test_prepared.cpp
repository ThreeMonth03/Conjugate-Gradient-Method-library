#include <cg/cg.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace
{

static_assert(!std::is_copy_constructible_v<cg::PreparedSolver>);
static_assert(std::is_nothrow_move_constructible_v<cg::PreparedSolver>);

TEST(PreparedSolver, DenseAndCsrOwnIndependentSnapshotsAndReuseWorkspace)
{
    for (const auto kind : {cg::PreconditionerKind::none, cg::PreconditionerKind::jacobi})
    {
        cg::DenseMatrix dense_input(2, 2, {4, 1, 1, 3});
        cg::CsrMatrix csr_input(2, 2, {0, 2, 4}, {0, 1, 0, 1}, {4, 1, 1, 3});
        const cg::PreparedSolver dense(dense_input, 1, kind);
        const cg::PreparedSolver csr(csr_input, 1, kind);
        dense_input = cg::DenseMatrix(1, 1, {8});
        csr_input = cg::CsrMatrix(1, 1, {0, 1}, {0}, {8});
        for (const auto *plan : {&dense, &csr})
        {
            ASSERT_EQ(plan->size(), 2u);
            EXPECT_EQ(plan->num_threads(), 1);
            EXPECT_EQ(plan->dense_kernel(), cg::DenseKernel::native);
            EXPECT_EQ(plan->preconditioner(), kind);
            EXPECT_EQ(plan->preconditioner_nbytes(), kind == cg::PreconditionerKind::none ? 0u : 2 * sizeof(double));
            if (kind == cg::PreconditionerKind::none)
            {
                EXPECT_EQ(plan->preconditioner_setup_ns(), 0u);
            }
            auto workspace = plan->create_workspace();
            const auto *storage = workspace.residual().data();
            const std::vector<double> b{1, 2};
            for (const bool fused : {false, true})
            {
                std::vector<double> x(2);
                const auto info = plan->solve_into(b, x, workspace, {}, fused);
                EXPECT_EQ(info.status, cg::Status::converged);
                EXPECT_NEAR(x[0], 1.0 / 11, 1e-12);
                EXPECT_NEAR(x[1], 7.0 / 11, 1e-12);
                EXPECT_EQ(workspace.residual().data(), storage);
            }
        }
    }
}

TEST(PreparedSolver, RejectsUnsupportedStateBeforeExposingAnObject)
{
    EXPECT_THROW(cg::PreparedSolver(cg::DenseMatrix(1, 2, {1, 2})), std::invalid_argument);
    EXPECT_THROW(cg::PreparedSolver(cg::CsrMatrix(1, 2, {0, 1}, {0}, {1})), std::invalid_argument);
    EXPECT_THROW(cg::PreparedSolver(cg::DenseMatrix(1, 1, {1}), 1, static_cast<cg::PreconditionerKind>(42)),
                 std::invalid_argument);
    EXPECT_THROW(cg::PreparedSolver(cg::CsrMatrix(1, 1, {0, 1}, {0}, {1}), 1, cg::PreconditionerKind::none,
                                    cg::DenseKernel::blas),
                 std::invalid_argument);
}

TEST(PreparedSolver, OneShotSetupPreservesBorrowedMatrixAndRejectsInvalidPolicyBeforeOutput)
{
    const cg::DenseMatrix A(2, 2, {4, 1, 1, 3});
    const std::vector<double> b{1, 2};
    std::vector<double> x(2);
    cg::Workspace workspace(2);
    const cg::CpuBackend backend;
    const auto info = cg::solve_into(A.view(), b, x, workspace, backend, {}, cg::PreconditionerKind::jacobi);
    EXPECT_EQ(info.status, cg::Status::converged);
    EXPECT_NEAR(x[0], 1.0 / 11, 1e-12);
    EXPECT_EQ(A.data()[0], 4);
    cg::CgOptions explicit_options;
    explicit_options.algorithm = cg::Algorithm::explicit_residual;
    const auto before = x;
    EXPECT_THROW(static_cast<void>(cg::solve_into(A.view(), b, x, workspace, backend, explicit_options,
                                                  cg::PreconditionerKind::jacobi)),
                 std::invalid_argument);
    EXPECT_EQ(x, before);
}

} // namespace
