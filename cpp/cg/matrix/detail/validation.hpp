#pragma once

#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/storage/detail/validation.hpp>

namespace cg
{

class CsrMatrix;

namespace detail
{
void validate(DenseMatrixView view);
bool overlaps(DenseMatrixView a, ConstVectorView b);
bool overlaps(const CsrMatrix &A, ConstVectorView x);
} // namespace detail

} // namespace cg
