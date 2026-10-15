#pragma once

#include <span>

namespace cg
{

using ConstVectorView = std::span<const double>;
using VectorView = std::span<double>;

} // namespace cg
