#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/storage/cuda/detail/validation.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void validate(DenseMatrixView view, const Context &context, bool inspect_pointer = true);
bool overlaps(DenseMatrixView A, ConstVectorView b);
bool overlaps(const CsrMatrix &A, ConstVectorView b);

} // namespace detail
} // namespace cuda
} // namespace cg
