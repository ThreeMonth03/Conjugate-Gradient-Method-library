#pragma once

#include <cg/matrix/cuda/csr_matrix_view.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void initialize_jacobi(DenseMatrixView A, VectorView inverse, const Context &context);
void initialize_jacobi(CsrMatrixView A, VectorView inverse, const Context &context);

} // namespace detail
} // namespace cuda
} // namespace cg
