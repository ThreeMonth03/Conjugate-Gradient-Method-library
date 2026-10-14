#include "cg/csr.hpp"

#include "cg/backend.hpp"

#include <limits>
#include <stdexcept>
#include <utility>

namespace cg
{
namespace
{

void validate_apply(const CsrMatrix &A, ConstVectorView x, VectorView output)
{
    detail::validate(x);
    detail::validate(output);
    if (A.cols() != x.size || A.rows() != output.size)
    {
        throw std::invalid_argument("CSR matrix-vector dimensions do not match");
    }
    if (detail::overlaps(x, output) || detail::overlaps(A, output))
    {
        throw std::invalid_argument("CSR output must be disjoint from inputs");
    }
}

} // namespace

CsrMatrix::CsrMatrix(std::size_t rows, std::size_t cols, std::vector<std::size_t> indptr,
                     std::vector<std::size_t> indices, std::vector<double> values)
    : rows_(rows), cols_(cols), indptr_(std::move(indptr)), indices_(std::move(indices)),
      values_(std::move(values))
{
    if (rows == std::numeric_limits<std::size_t>::max() || indptr_.size() != rows + 1 ||
        indices_.size() != values_.size() || indptr_.front() != 0 ||
        indptr_.back() != values_.size())
    {
        throw std::invalid_argument("invalid CSR storage dimensions or endpoints");
    }
    for (std::size_t row = 0; row < rows; ++row)
    {
        const auto begin = indptr_[row];
        const auto end = indptr_[row + 1];
        if (begin > end || end > values_.size())
        {
            throw std::invalid_argument("CSR row pointers must be monotone and within storage");
        }
        for (std::size_t entry = begin; entry < end; ++entry)
        {
            if (indices_[entry] >= cols ||
                (entry != begin && indices_[entry - 1] >= indices_[entry]))
            {
                throw std::invalid_argument("CSR columns must be in range, sorted, and unique");
            }
        }
    }
}

CsrMatrix::CsrMatrix(CsrMatrix &&other) noexcept
    : rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)),
      indptr_(std::move(other.indptr_)), indices_(std::move(other.indices_)),
      values_(std::move(other.values_))
{
}

CsrMatrix &CsrMatrix::operator=(CsrMatrix &&other) noexcept
{
    if (this != &other)
    {
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        indptr_ = std::move(other.indptr_);
        indices_ = std::move(other.indices_);
        values_ = std::move(other.values_);
    }
    return *this;
}

namespace detail
{

bool overlaps(const CsrMatrix &A, ConstVectorView x)
{
    return overlaps(ConstVectorView{A.values().data(), A.nnz()}, x);
}

} // namespace detail

void CpuBackend::apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const
{
    if (dense_kernel() == DenseKernel::blas)
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    validate_apply(A, x, output);
    const auto *indptr = A.indptr().data();
    const auto *indices = A.indices().data();
    const auto *values = A.values().data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)                                                          \
    num_threads(threads_) if (threads_ > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        double sum = 0.0;
        for (std::size_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            sum += values[entry] * x.data[indices[entry]];
        }
        output.data[row] = sum;
    }
}

double CpuBackend::apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const
{
    if (dense_kernel() == DenseKernel::blas)
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    validate_apply(A, p, ap);
    if (p.size != ap.size)
    {
        throw std::invalid_argument("CSR apply_dot requires a square matrix");
    }
    if (!fused_)
    {
        apply(A, p, ap);
        return dot(p, ap);
    }
    const auto *indptr = A.indptr().data();
    const auto *indices = A.indices().data();
    const auto *values = A.values().data();
    double product = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : product) schedule(static)                                   \
    num_threads(threads_) if (threads_ > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        double sum = 0.0;
        for (std::size_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            sum += values[entry] * p.data[indices[entry]];
        }
        ap.data[row] = sum;
        product += p.data[row] * sum;
    }
    return product;
}

double CpuBackend::residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                            bool b_minus_ax) const
{
    if (dense_kernel() == DenseKernel::blas)
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    detail::validate(b);
    validate_apply(A, x, r);
    if (b.size != r.size || detail::overlaps(b, r))
    {
        throw std::invalid_argument("CSR residual vectors must match and be disjoint");
    }
    if (!fused_)
    {
        apply(A, x, r);
        axpby(b_minus_ax ? 1.0 : -1.0, b, b_minus_ax ? -1.0 : 1.0, r, r);
        return norm2(r);
    }
    const auto *indptr = A.indptr().data();
    const auto *indices = A.indices().data();
    const auto *values = A.values().data();
    double squared_norm = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : squared_norm) schedule(static)                              \
    num_threads(threads_) if (threads_ > 1 && A.rows() >= 4096)
#endif
    for (std::size_t row = 0; row < A.rows(); ++row)
    {
        double sum = 0.0;
        for (std::size_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            sum += values[entry] * x.data[indices[entry]];
        }
        const double value = b_minus_ax ? b.data[row] - sum : sum - b.data[row];
        r.data[row] = value;
        squared_norm += value * value;
    }
    return squared_norm;
}

} // namespace cg
