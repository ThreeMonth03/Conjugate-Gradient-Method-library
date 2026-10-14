#include <_cg_method.hpp>
#include <_matrix.hpp>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

TEST(LegacyMatrix, RectangularMatrixVectorProduct)
{
    Matrix::Naive_Matrix matrix(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
    Matrix::Naive_Matrix vector(std::vector<double>{2, -1, 3});
    auto product = matrix * vector;
    ASSERT_EQ(product.nrow(), 2);
    ASSERT_EQ(product.ncol(), 1);
    EXPECT_DOUBLE_EQ(product(0, 0), 9);
    EXPECT_DOUBLE_EQ(product(1, 0), 21);
}

TEST(LegacyCG, SolvesFromZeroInitialGuess)
{
    Matrix::Naive_Matrix matrix(2, 2, std::vector<double>{4, 1, 1, 3});
    Matrix::Naive_Matrix rhs(std::vector<double>{1, 2});
    Matrix::Naive_Matrix initial(2, 1);
    cg_method::linear_CG solver(1e-12, 20, 1);
    auto solution = solver.solve_by_Naive_Matrix(matrix, rhs, initial);
    EXPECT_NEAR(solution(0, 0), 1.0 / 11.0, 1e-12);
    EXPECT_NEAR(solution(1, 0), 7.0 / 11.0, 1e-12);
    EXPECT_LT((matrix * solution - rhs).norm(), 1e-12);
    EXPECT_DOUBLE_EQ(initial.norm(), 0);
}

template <class MatrixType> class LegacyStorage : public ::testing::Test
{
};

using MatrixTypes = ::testing::Types<Matrix::Naive_Matrix, Matrix::Accelerated_Matrix>;
TYPED_TEST_SUITE(LegacyStorage, MatrixTypes);

TYPED_TEST(LegacyStorage, EmptyShapesRemainUsableAfterCopyMoveAndAssignment)
{
    TypeParam empty(0, 3);
    TypeParam copy(empty);
    EXPECT_EQ(copy.nrow(), 0);
    EXPECT_EQ(copy.ncol(), 3);
    EXPECT_TRUE(copy.buffer_vector().empty());

    TypeParam rows(std::vector<std::vector<double>>{{}, {}, {}});
    EXPECT_EQ(rows.nrow(), 3);
    EXPECT_EQ(rows.ncol(), 0);
    EXPECT_EQ(rows.buffer_vector2d().size(), 3);
    EXPECT_TRUE(rows.buffer_vector().empty());

    TypeParam assigned(2, 2);
    assigned = empty;
    EXPECT_EQ(assigned.nrow(), 0);
    EXPECT_EQ(assigned.ncol(), 3);
    EXPECT_DOUBLE_EQ(assigned.norm(), 0);
    assigned = std::vector<double>{};

    TypeParam moved(std::move(rows));
    EXPECT_EQ(moved.nrow(), 3);
    EXPECT_EQ(moved.ncol(), 0);
    EXPECT_EQ(rows.nrow(), 0);
    EXPECT_EQ(rows.ncol(), 0);
    EXPECT_TRUE(rows.buffer_vector().empty());

    assigned = TypeParam(std::vector<double>{1, 2});
    EXPECT_DOUBLE_EQ(assigned(0, 0), 1);
    EXPECT_DOUBLE_EQ(assigned(1, 0), 2);
}

TYPED_TEST(LegacyStorage, RejectsElementAndByteSizeOverflow)
{
    const size_t maximum = std::numeric_limits<size_t>::max();
    EXPECT_THROW((TypeParam(maximum / 2 + 1, 2)), std::length_error);
    EXPECT_THROW((TypeParam(maximum / sizeof(double) + 1, 1)), std::length_error);
    EXPECT_THROW((TypeParam(maximum / 2 + 1, 2, std::vector<double>{})), std::length_error);
}
