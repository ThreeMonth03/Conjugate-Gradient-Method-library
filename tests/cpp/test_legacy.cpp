#include <_cg_method.hpp>
#include <_matrix.hpp>
#include <gtest/gtest.h>
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
