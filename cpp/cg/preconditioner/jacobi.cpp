#include <cg/matrix/validation.hpp>
#include <cg/preconditioner/jacobi.hpp>

#include <cg/matrix/csr_matrix.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace cg
{
namespace
{

double checked_inverse(double diagonal)
{
    if (!std::isfinite(diagonal) || diagonal <= 0.0)
    {
        throw std::invalid_argument("Jacobi requires a finite strictly positive diagonal");
    }
    const double inverse = 1.0 / diagonal;
    if (!std::isfinite(inverse) || inverse <= 0.0)
    {
        throw std::invalid_argument("Jacobi inverse diagonal must be finite and positive");
    }
    return inverse;
}

std::vector<double> dense_inverse(DenseMatrixView A)
{
    detail::validate(A);
    if (A.rows == 0 || A.rows != A.cols)
    {
        throw std::invalid_argument("Jacobi requires a nonempty square matrix");
    }
    std::vector<double> inverse(A.rows);
    for (std::size_t row = 0; row < A.rows; ++row)
    {
        inverse[row] = checked_inverse(A.data[row * A.row_stride + row]);
    }
    return inverse;
}

std::vector<double> csr_inverse(const CsrMatrix &A)
{
    if (A.rows() == 0 || A.rows() != A.cols())
    {
        throw std::invalid_argument("Jacobi requires a nonempty square matrix");
    }
    std::vector<double> inverse(A.rows());
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        const auto begin = A.indices().begin() + A.indptr()[row];
        const auto end = A.indices().begin() + A.indptr()[row + 1];
        const auto diagonal = std::lower_bound(begin, end, row);
        if (diagonal == end || *diagonal != row)
        {
            throw std::invalid_argument("Jacobi requires a stored diagonal entry in every CSR row");
        }
        const auto entry = static_cast<std::size_t>(diagonal - A.indices().begin());
        inverse[row] = checked_inverse(A.values()[entry]);
    }
    return inverse;
}

void validate_output(ConstVectorView inverse, VectorView z)
{
    if (detail::overlaps(inverse, z))
    {
        throw std::invalid_argument("Jacobi output must be disjoint from its inverse diagonal");
    }
}

} // namespace

JacobiPreconditioner::JacobiPreconditioner(DenseMatrixView A) : m_inverse(dense_inverse(A)) {}
JacobiPreconditioner::JacobiPreconditioner(const CsrMatrix &A) : m_inverse(csr_inverse(A)) {}

void JacobiPreconditioner::apply(ConstVectorView r, VectorView z, const CpuBackend &backend) const
{
    validate_output(m_inverse.view(), z);
    backend.multiply(m_inverse.view(), r, z);
}

double JacobiPreconditioner::apply_dot(ConstVectorView r, VectorView z, const CpuBackend &backend) const
{
    validate_output(m_inverse.view(), z);
    return backend.multiply_dot(m_inverse.view(), r, z);
}

} // namespace cg
