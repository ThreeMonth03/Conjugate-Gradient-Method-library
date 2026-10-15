#include <cg/matrix/cuda/detail/validation.hpp>

namespace cg
{
namespace cuda
{
namespace detail
{

void validate(DenseMatrixView view, const Context &context, bool inspect_pointer)
{
    validate(ConstVectorView{view.data(), storage_extent(view), view.device()}, context, inspect_pointer);
}

bool overlaps(DenseMatrixView A, ConstVectorView b)
{
    return overlaps(ConstVectorView{A.data(), storage_extent(A), A.device()}, b);
}

bool overlaps(const CsrMatrix &A, ConstVectorView b)
{
    const auto view = A.view();
    const auto pointer_overlap = [&](const void *pointer, std::size_t count)
    { return overlaps(ConstVectorView{static_cast<const double *>(pointer), count, view.device()}, b); };
    return pointer_overlap(view.data(), view.nnz()) || pointer_overlap(view.indptr(), view.rows() + 1) ||
           pointer_overlap(view.indices(), view.nnz());
}

} // namespace detail
} // namespace cuda
} // namespace cg
