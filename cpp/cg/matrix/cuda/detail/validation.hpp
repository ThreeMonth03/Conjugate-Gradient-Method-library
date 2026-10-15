#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/storage/cuda/detail/validation.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

inline std::size_t storage_extent(DenseMatrixView A)
{
    if (A.row_stride() < A.cols())
    {
        throw std::invalid_argument("CUDA row stride is smaller than its columns");
    }
    if (A.rows() == 0 || A.cols() == 0)
    {
        return 0;
    }
    const auto preceding = checked_product(A.rows() - 1, A.row_stride());
    if (preceding > std::numeric_limits<std::size_t>::max() - A.cols())
    {
        throw std::invalid_argument("CUDA matrix span overflows size_t");
    }
    return preceding + A.cols();
}
void validate(DenseMatrixView view, const Context &context, bool inspect_pointer = true);
bool overlaps(DenseMatrixView A, ConstVectorView b);
bool overlaps(const CsrMatrix &A, ConstVectorView b);

} // namespace detail
} // namespace cuda
} // namespace cg
