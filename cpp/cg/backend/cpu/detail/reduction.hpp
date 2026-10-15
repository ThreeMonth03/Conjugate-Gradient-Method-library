#pragma once

#include <cg/storage/vector_view.hpp>

namespace cg
{
namespace detail
{

// The view was validated by the operation that produced squared_norm.
double norm_from_squared(ConstVectorView x, double squared_norm);

} // namespace detail
} // namespace cg
