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

TEST(LegacyMatrix, RectangularHelpersMatchTheKnownProduct)
{
    const Matrix::Naive_Matrix left(2, 3, {1, 2, 3, 4, 5, 6});
    const Matrix::Naive_Matrix right(3, 4, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    const Matrix::Naive_Matrix expected(2, 4, {38, 44, 50, 56, 83, 98, 113, 128});
    EXPECT_EQ(Matrix::multiply_naive(left, right), expected);
    for (const std::size_t tile : {1u, 2u, 4u, 100u})
    {
        EXPECT_EQ(Matrix::multiply_tile(left, right, tile), expected);
    }
    EXPECT_THROW(Matrix::multiply_tile(left, right, 0), std::invalid_argument);
}

TEST(LegacyMatrix, ScalarMultiplyPreservesRectangularAndEmptyShapes)
{
    Matrix::Naive_Matrix source(2, 3, {1, 2, 3, 4, 5, 6});
    const auto result = source * 0.25;
    EXPECT_EQ(result.nrow(), 2u);
    EXPECT_EQ(result.ncol(), 3u);
    EXPECT_DOUBLE_EQ(result(0, 2), 0.75);
    EXPECT_DOUBLE_EQ(result(1, 2), 1.5);
    EXPECT_DOUBLE_EQ(source(1, 2), 6.0);
    Matrix::Naive_Matrix empty(3, 0);
    const auto empty_result = empty * 2.0;
    EXPECT_EQ(empty_result.nrow(), 3u);
    EXPECT_EQ(empty_result.ncol(), 0u);
    EXPECT_TRUE(empty_result.buffer_vector().empty());
}

TEST(LegacyMatrix, AllAcceleratedMultiplyPathsRejectInvalidThreadCounts)
{
    for (const int invalid : {0, -1})
    {
        Matrix::Accelerated_Matrix matrix(2, 2, {1, 0, 0, 1});
        Matrix::Accelerated_Matrix scalar(1, 1, {2});
        matrix.set_number_of_threads() = invalid;
        EXPECT_THROW(matrix * 2.0, std::invalid_argument);
        EXPECT_THROW(matrix * scalar, std::invalid_argument);
        EXPECT_THROW(matrix * matrix, std::invalid_argument);
        matrix.set_number_of_threads() = 1;
        scalar.set_number_of_threads() = invalid;
        EXPECT_THROW(scalar * matrix, std::invalid_argument);
        scalar.set_number_of_threads() = 1;
        EXPECT_DOUBLE_EQ((scalar * matrix)(0, 0), 2);
    }
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

TEST(LegacyCG, NumericalFailureIsNotReturnedAsAnOrdinarySolution)
{
    Matrix::Naive_Matrix identity(2, 2, std::vector<double>{1, 0, 0, 1});
    Matrix::Naive_Matrix tiny_rhs(std::vector<double>{1e-200, -1e-200});
    Matrix::Naive_Matrix initial(2, 1);
    cg_method::linear_CG solver(1e-208, 20, 1);
    EXPECT_THROW(solver.solve_by_Naive_Matrix(identity, tiny_rhs, initial), std::runtime_error);
    EXPECT_DOUBLE_EQ(initial.norm(), 0.0);
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
