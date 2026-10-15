#pragma once

#include <cg/matrix/dense_matrix_view.hpp>
#include <cg/storage/detail/validation.hpp>

namespace cg
{

namespace detail
{
void validate(DenseMatrixView view);
bool overlaps(DenseMatrixView a, ConstVectorView b);
} // namespace detail

} // namespace cg
