#pragma once

#include <cg/storage/detail/validation.hpp>

#include <stdexcept>

namespace cg
{
namespace detail
{

inline void same_size(ConstVectorView x, ConstVectorView y)
{
    detail::validate(x);
    detail::validate(y);
    if (x.size() != y.size())
    {
        throw std::invalid_argument("vector dimensions do not match");
    }
}

inline void no_overlap(ConstVectorView x, ConstVectorView y)
{
    if (detail::overlaps(x, y))
    {
        throw std::invalid_argument("operation requires disjoint storage");
    }
}

inline void exact_or_disjoint(ConstVectorView input, ConstVectorView output)
{
    if (input.data() != output.data())
    {
        no_overlap(input, output);
    }
}

} // namespace detail
} // namespace cg
