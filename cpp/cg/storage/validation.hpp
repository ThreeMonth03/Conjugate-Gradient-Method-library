#pragma once

#include <cg/storage/vector_view.hpp>

namespace cg
{

namespace detail
{
std::size_t checked_product(std::size_t a, std::size_t b);
void validate(ConstVectorView view);
bool overlaps(ConstVectorView a, ConstVectorView b);
} // namespace detail

} // namespace cg
