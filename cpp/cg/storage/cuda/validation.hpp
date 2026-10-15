#pragma once

#include <cg/backend/cuda/context.hpp>
#include <cg/storage/cuda/vector_view.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void validate(ConstVectorView view, const Context &context, bool inspect_pointer = true);
void validate_pointer(const void *data, std::size_t bytes, std::size_t alignment, int device, const Context &context,
                      bool inspect_pointer = true);
bool overlaps(ConstVectorView a, ConstVectorView b);
void fill(VectorView output, double value, const Context &context);

} // namespace detail
} // namespace cuda
} // namespace cg
