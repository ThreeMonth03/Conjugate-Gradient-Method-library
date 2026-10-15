#include <cg/matrix/dense_matrix.hpp>
#include <cg/matrix/detail/storage_extent.hpp>
#include <cg/matrix/detail/validation.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace
{

constexpr auto max_size = std::numeric_limits<std::size_t>::max();
static_assert(cg::detail::checked_product(max_size, 0) == 0);
static_assert(cg::detail::checked_product(0, max_size) == 0);
static_assert(cg::detail::checked_product(max_size, 1) == max_size);
static_assert(cg::detail::storage_extent(0, max_size, max_size) == 0);
static_assert(cg::detail::storage_extent(max_size, 0, max_size) == 0);
static_assert(cg::detail::storage_extent(1, max_size, max_size) == max_size);
static_assert(cg::detail::storage_extent(3, 2, 4) == 10);

TEST(StorageSizes, CheckedProductRejectsOnlyUnrepresentableResults)
{
    EXPECT_EQ(cg::detail::checked_product(max_size / 2, 2), max_size - 1);
    EXPECT_THROW(cg::detail::checked_product(max_size, 2), std::invalid_argument);
    EXPECT_THROW(cg::detail::checked_product(2, max_size), std::invalid_argument);
    EXPECT_THROW(cg::detail::checked_product(max_size / 2 + 1, 2), std::invalid_argument);
}

TEST(StorageSizes, ExtentValidatesStrideBeforeAcceptingEmptyShapes)
{
    EXPECT_THROW(cg::detail::storage_extent(0, 2, 1), std::invalid_argument);
    EXPECT_THROW(cg::detail::storage_extent(3, 2, 1), std::invalid_argument);
    EXPECT_EQ(cg::detail::storage_extent(0, 2, 2), 0u);
    EXPECT_EQ(cg::detail::storage_extent(3, 0, max_size), 0u);
    EXPECT_EQ(cg::detail::storage_extent(2, 2, 3), 5u);
    EXPECT_EQ(cg::detail::storage_extent(1, 2, max_size), 2u);
}

TEST(StorageSizes, ExtentChecksBothStrideProductAndFinalRowAddition)
{
    EXPECT_EQ(cg::detail::storage_extent(2, 1, max_size - 1), max_size);
    EXPECT_THROW(cg::detail::storage_extent(2, 1, max_size), std::invalid_argument);
    EXPECT_THROW(cg::detail::storage_extent(3, 1, max_size), std::invalid_argument);
}

TEST(StorageSizes, DenseViewOverlapIncludesPaddingBetweenRowsButNotTrailingPadding)
{
    std::array<double, 6> storage{};
    const cg::DenseMatrixView matrix(storage.data(), 2, 2, 3);
    EXPECT_TRUE(cg::detail::overlaps(matrix, cg::ConstVectorView{storage.data() + 2, 1}));
    EXPECT_TRUE(cg::detail::overlaps(matrix, cg::ConstVectorView{storage.data() + 4, 1}));
    EXPECT_FALSE(cg::detail::overlaps(matrix, cg::ConstVectorView{storage.data() + 5, 1}));
}

TEST(StorageSizes, DenseViewsValidateByteCountsAndAddressesAfterComputingExtent)
{
    EXPECT_THROW(cg::DenseMatrix(2, max_size / 2 + 1), std::invalid_argument);
    double value = 0.0;
    EXPECT_THROW((cg::DenseMatrixView(&value, 1, max_size)), std::invalid_argument);
    const auto address = std::numeric_limits<std::uintptr_t>::max() & ~(alignof(double) - 1);
    const auto *near_end = reinterpret_cast<const double *>(address);
    EXPECT_THROW((cg::DenseMatrixView(near_end, 1, 1)), std::invalid_argument);
    EXPECT_NO_THROW((cg::DenseMatrixView(nullptr, 0, 2)));
    EXPECT_NO_THROW((cg::DenseMatrixView(nullptr, 3, 0, max_size)));
}

} // namespace
