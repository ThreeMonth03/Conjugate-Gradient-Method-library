#include <cg/cg.hpp>
#include <support/allocation_failure.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <new>
#include <utility>
#include <vector>

namespace
{

TEST(MatrixAssignment, DenseAllocationFailurePreservesShapeStorageAndValues)
{
    const cg::DenseMatrix source(2, 2, {4, 1, 1, 3});
    cg::DenseMatrix target(1, 1, {7});
    const auto *storage = target.data();
    bool threw = false;
    {
        const cg::test::AllocationFailureGuard failure(0);
        try
        {
            target = source;
        }
        catch (const std::bad_alloc &)
        {
            threw = true;
        }
    }
    ASSERT_TRUE(threw);
    EXPECT_EQ(target.rows(), 1u);
    EXPECT_EQ(target.cols(), 1u);
    EXPECT_EQ(target.data(), storage);
    EXPECT_DOUBLE_EQ(target.data()[0], 7);
    std::vector<double> input{2}, output(1);
    cg::CpuBackend().apply(target.view(), input, output);
    EXPECT_DOUBLE_EQ(output[0], 14);
    EXPECT_EQ(source.rows(), 2u);
    EXPECT_DOUBLE_EQ(source.data()[3], 3);

    target = source;
    EXPECT_EQ(target.rows(), 2u);
    EXPECT_DOUBLE_EQ(target.data()[3], 3);
    target.data()[0] = 9;
    EXPECT_DOUBLE_EQ(source.data()[0], 4);
}

TEST(MatrixAssignment, CsrFailureAtEveryCopiedArrayPreservesCanonicalStorage)
{
    const cg::CsrMatrix source(2, 2, {0, 2, 4}, {0, 1, 0, 1}, {4, 1, 1, 3});
    for (int allocation = 0; allocation < 3; ++allocation)
    {
        cg::CsrMatrix target(1, 1, {0, 1}, {0}, {7});
        const auto *indptr = target.indptr().data();
        const auto *indices = target.indices().data();
        const auto *values = target.values().data();
        bool threw = false;
        {
            const cg::test::AllocationFailureGuard failure(allocation);
            try
            {
                target = source;
            }
            catch (const std::bad_alloc &)
            {
                threw = true;
            }
        }
        ASSERT_TRUE(threw) << allocation;
        EXPECT_EQ(target.rows(), 1u);
        EXPECT_EQ(target.cols(), 1u);
        EXPECT_EQ(target.nnz(), 1u);
        EXPECT_EQ(target.indptr().data(), indptr);
        EXPECT_EQ(target.indices().data(), indices);
        EXPECT_EQ(target.values().data(), values);
        EXPECT_TRUE(std::ranges::equal(target.indptr(), std::vector<std::size_t>{0, 1}));
        EXPECT_TRUE(std::ranges::equal(target.indices(), std::vector<std::size_t>{0}));
        EXPECT_TRUE(std::ranges::equal(target.values(), std::vector<double>{7}));
        std::vector<double> input{2}, output(1);
        cg::CpuBackend().apply(target, input, output);
        EXPECT_DOUBLE_EQ(output[0], 14);
        target = source;
        EXPECT_EQ(target.rows(), 2u);
        EXPECT_TRUE(std::ranges::equal(target.indptr(), source.indptr()));
        EXPECT_TRUE(std::ranges::equal(target.indices(), source.indices()));
        EXPECT_TRUE(std::ranges::equal(target.values(), source.values()));
        EXPECT_NE(target.values().data(), source.values().data());
    }
}

TEST(MatrixAssignment, SelfCopyDoesNotAllocateAndSwapPreservesCompleteValues)
{
    cg::DenseMatrix dense(1, 1, {7}), other_dense(2, 2, {4, 1, 1, 3});
    cg::CsrMatrix sparse(1, 1, {0, 1}, {0}, {7});
    cg::CsrMatrix other_sparse(2, 2, {0, 1, 2}, {0, 1}, {4, 3});
    auto &dense_alias = dense;
    auto &sparse_alias = sparse;
    {
        const cg::test::AllocationFailureGuard failure(0);
        dense = dense_alias;
        sparse = sparse_alias;
        dense.swap(other_dense);
        sparse.swap(other_sparse);
    }
    EXPECT_EQ(dense.rows(), 2u);
    EXPECT_DOUBLE_EQ(dense.data()[3], 3);
    EXPECT_EQ(other_dense.rows(), 1u);
    EXPECT_DOUBLE_EQ(other_dense.data()[0], 7);
    EXPECT_EQ(sparse.rows(), 2u);
    EXPECT_TRUE(std::ranges::equal(sparse.values(), std::vector<double>{4, 3}));
    EXPECT_EQ(other_sparse.rows(), 1u);
    EXPECT_TRUE(std::ranges::equal(other_sparse.values(), std::vector<double>{7}));
}

} // namespace
