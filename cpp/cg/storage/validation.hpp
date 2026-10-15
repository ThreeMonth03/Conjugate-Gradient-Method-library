#pragma once

#include <cg/storage/vector_view.hpp>

namespace cg
{

namespace detail
{
std::size_t checked_product(std::size_t a, std::size_t b);
// Check foreign pointer/count metadata before constructing a span. The caller
// still owns the contract for the actual allocation's extent and lifetime.
void validate_raw_vector(const double *data, std::size_t size);
bool overlaps_raw(const double *a, std::size_t a_size, const double *b, std::size_t b_size);
void validate(ConstVectorView view);
bool overlaps(ConstVectorView a, ConstVectorView b);
} // namespace detail

} // namespace cg
