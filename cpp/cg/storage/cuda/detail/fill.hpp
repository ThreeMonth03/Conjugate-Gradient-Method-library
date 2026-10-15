#pragma once

#include <cg/runtime/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void fill(VectorView output, double value, const Context &context);

} // namespace detail
} // namespace cuda
} // namespace cg
