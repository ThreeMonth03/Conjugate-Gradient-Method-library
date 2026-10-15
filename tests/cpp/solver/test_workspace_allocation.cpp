#include <cg/cg.hpp>
#include <support/allocation_failure.hpp>

#include <gtest/gtest.h>

#include <array>
#include <new>
#include <vector>

namespace
{

std::array<cg::VectorView, 5> buffers(cg::Workspace &workspace)
{
    return {workspace.residual(), workspace.direction(), workspace.product(), workspace.scratch(),
            workspace.preconditioned()};
}

TEST(WorkspaceAllocation, FailureDuringGrowthPreservesEveryBufferAndValues)
{
    for (int allocation = 0; allocation < 5; ++allocation)
    {
        cg::Workspace workspace(1);
        const auto original = buffers(workspace);
        const auto capacity = workspace.capacity();
        for (std::size_t index = 0; index < original.size(); ++index)
        {
            original[index][0] = static_cast<double>(index + 1);
        }
        bool threw = false;
        {
            const cg::test::AllocationFailureGuard failure(allocation);
            try
            {
                workspace.prepare(capacity + 1);
            }
            catch (const std::bad_alloc &)
            {
                threw = true;
            }
        }
        ASSERT_TRUE(threw) << allocation;
        EXPECT_EQ(workspace.size(), 1u);
        EXPECT_EQ(workspace.capacity(), capacity);
        const auto preserved = buffers(workspace);
        for (std::size_t index = 0; index < preserved.size(); ++index)
        {
            EXPECT_EQ(preserved[index].size(), 1u);
            EXPECT_EQ(preserved[index].data(), original[index].data());
            EXPECT_DOUBLE_EQ(preserved[index][0], static_cast<double>(index + 1));
        }
        const cg::DenseMatrix matrix(1, 1, {2});
        const std::vector<double> b{4};
        std::vector<double> x{0};
        const auto info = cg::solve_into(matrix.view(), b, x, workspace, cg::CpuBackend());
        EXPECT_EQ(info.status, cg::Status::converged);
        EXPECT_DOUBLE_EQ(x[0], 2);
        workspace.prepare(capacity + 1);
        for (const auto buffer : buffers(workspace))
        {
            EXPECT_EQ(buffer.size(), capacity + 1);
        }
    }
}

TEST(WorkspaceAllocation, ShrinkingAndRegrowingWithinCapacityNeverAllocates)
{
    cg::Workspace workspace(32);
    const auto original = buffers(workspace);
    bool threw = false;
    {
        const cg::test::AllocationFailureGuard failure(0);
        try
        {
            workspace.prepare(32);
            workspace.prepare(5);
            workspace.prepare(0);
            workspace.prepare(32);
        }
        catch (const std::bad_alloc &)
        {
            threw = true;
        }
    }
    ASSERT_FALSE(threw);
    const auto reused = buffers(workspace);
    for (std::size_t index = 0; index < reused.size(); ++index)
    {
        EXPECT_EQ(reused[index].size(), 32u);
        EXPECT_EQ(reused[index].data(), original[index].data());
    }
}

} // namespace
