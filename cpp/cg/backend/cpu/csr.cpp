#include <cg/backend/cpu/backend.hpp>
#include <cg/matrix/csr_matrix.hpp>
#include <cg/storage/detail/validation.hpp>

#include <span>
#include <stdexcept>

namespace cg
{

namespace
{
inline double row_dot(std::span<const std::size_t> indices, ConstVectorView values, ConstVectorView x,
                      std::size_t begin, std::size_t end)
{
    double sum = 0.0;
    for (std::size_t entry = begin; entry < end; ++entry)
    {
        sum += values[entry] * x[indices[entry]];
    }
    return sum;
}

void validate_apply(const CsrMatrix &A, ConstVectorView x, VectorView output)
{
    detail::validate(x);
    detail::validate(output);
    if (A.cols() != x.size() || A.rows() != output.size())
    {
        throw std::invalid_argument("CSR matrix-vector dimensions do not match");
    }
    if (detail::overlaps(x, output) || detail::overlaps(A, output))
    {
        throw std::invalid_argument("CSR output must be disjoint from inputs");
    }
}

} // namespace
void CpuBackend::apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const
{
    if (dense_kernel() == DenseKernel::blas)
    {
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    }
    validate_apply(A, x, output);
    const auto indptr = A.indptr();
    const auto indices = A.indices();
    const auto values = A.values();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        const double sum = row_dot(indices, values, x, indptr[row], indptr[row + 1]);
        output[row] = sum;
    }
}

double CpuBackend::apply_and_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const
{
    if (dense_kernel() == DenseKernel::blas)
    {
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    }
    validate_apply(A, p, ap);
    if (p.size() != ap.size())
    {
        throw std::invalid_argument("CSR apply_and_dot requires a square matrix");
    }
    if (!m_fused)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    const auto indptr = A.indptr();
    const auto indices = A.indices();
    const auto values = A.values();
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                                       \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        const double sum = row_dot(indices, values, p, indptr[row], indptr[row + 1]);
        ap[row] = sum;
        product += p[row] * sum;
    }
    return product;
}

double CpuBackend::residual_squared_norm(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                                         ResidualSign sign) const
{
    const bool b_minus_ax = sign == ResidualSign::rhs_minus_product;
    if (dense_kernel() == DenseKernel::blas)
    {
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    }
    detail::validate(b);
    validate_apply(A, x, r);
    if (b.size() != r.size() || detail::overlaps(b, r))
    {
        throw std::invalid_argument("CSR residual vectors must match and be disjoint");
    }
    if (!m_fused)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return squared_norm(r);
    }
    const auto indptr = A.indptr();
    const auto indices = A.indices();
    const auto values = A.values();
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                                                  \
    num_threads(m_num_threads) if (m_num_threads > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        const double sum = row_dot(indices, values, x, indptr[row], indptr[row + 1]);
        const double value = b_minus_ax ? b[row] - sum : sum - b[row];
        r[row] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

} // namespace cg
