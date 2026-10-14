#ifndef CG_DETAIL_BLAS_HPP
#define CG_DETAIL_BLAS_HPP

#include "cg/storage.hpp"

namespace cg
{
namespace detail
{

// Caller validates dimensions and storage disjointness. The provider adapter
// checks the LP64 integer range before touching matrix or vector data.
void blas_apply(DenseMatrixView A, ConstVectorView x, VectorView output);

} // namespace detail
} // namespace cg

#endif
