#pragma once

#include <cg/matrix/cuda/detail/validation.hpp>

#include <stdexcept>

namespace cg
{
namespace cuda
{
namespace detail
{

inline void same_size(ConstVectorView a, ConstVectorView b, const Context &context)
{
    detail::validate(a, context, false);
    detail::validate(b, context, false);
    if (a.size() != b.size())
    {
        throw std::invalid_argument("CUDA vector dimensions do not match");
    }
}

inline void no_overlap(ConstVectorView a, ConstVectorView b)
{
    if (detail::overlaps(a, b))
    {
        throw std::invalid_argument("CUDA operation requires disjoint storage");
    }
}

inline void exact_or_disjoint(ConstVectorView a, ConstVectorView b)
{
    if (a.data() != b.data())
    {
        no_overlap(a, b);
    }
}

inline void validate_apply(DenseMatrixView A, ConstVectorView x, VectorView output, const Context &context)
{
    detail::validate(A, context, false);
    detail::validate(x, context, false);
    detail::validate(output, context, false);
    if (A.cols() != x.size() || A.rows() != output.size())
    {
        throw std::invalid_argument("CUDA matrix-vector dimensions do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("CUDA matrix and output storage overlap");
    }
}

inline void validate_apply(const CsrMatrix &A, ConstVectorView x, VectorView output, const Context &context)
{
    detail::validate(x, context, false);
    detail::validate(output, context, false);
    if (A.device() != context.device() || A.cols() != x.size() || A.rows() != output.size())
    {
        throw std::invalid_argument("CUDA CSR dimensions or device do not match");
    }
    no_overlap(x, output);
    if (detail::overlaps(A, output))
    {
        throw std::invalid_argument("CUDA CSR and output storage overlap");
    }
}

} // namespace detail
} // namespace cuda
} // namespace cg
