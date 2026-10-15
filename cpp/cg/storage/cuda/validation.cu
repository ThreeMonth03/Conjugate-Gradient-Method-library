#include <cg/matrix/cuda/validation.hpp>

namespace cg
{
namespace cuda
{

namespace detail
{

void validate_pointer(const void *data, std::size_t bytes, std::size_t alignment, int device, const Context &context,
                      bool inspect_pointer)
{
    if (bytes == 0)
    {
        return;
    }
    if (device != context.device())
    {
        throw std::invalid_argument("CUDA storage belongs to a different device");
    }
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || begin % alignment != 0 || begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("invalid CUDA storage pointer, alignment or span");
    }
    if (inspect_pointer)
    {
        DeviceGuard guard(device);
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, data);
        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            throw std::invalid_argument("expected CUDA device storage");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
        {
            throw std::invalid_argument("expected storage on the selected CUDA device");
        }
    }
}
void validate(ConstVectorView view, const Context &context, bool inspect_pointer)
{
    validate_pointer(view.data, product(view.size, sizeof(double)), alignof(double), view.device, context,
                     inspect_pointer);
}
void validate(DenseMatrixView view, const Context &context, bool inspect_pointer)
{
    validate(ConstVectorView{view.data, span(view), view.device}, context, inspect_pointer);
}
bool overlaps(ConstVectorView a, ConstVectorView b)
{
    if (a.size == 0 || b.size == 0 || a.device != b.device)
    {
        return false;
    }
    const auto abegin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto bbegin = reinterpret_cast<std::uintptr_t>(b.data);
    const auto abytes = product(a.size, sizeof(double)), bbytes = product(b.size, sizeof(double));
    if (abegin > std::numeric_limits<std::uintptr_t>::max() - abytes ||
        bbegin > std::numeric_limits<std::uintptr_t>::max() - bbytes)
    {
        throw std::invalid_argument("CUDA storage address span overflows uintptr_t");
    }
    return abegin < bbegin + bbytes && bbegin < abegin + abytes;
}
bool overlaps(DenseMatrixView A, ConstVectorView b)
{
    return overlaps(ConstVectorView{A.data, span(A), A.device}, b);
}
bool overlaps(const CsrMatrix &A, ConstVectorView b)
{
    const auto view = A.view();
    const auto pointer_overlap = [&](const void *pointer, std::size_t count)
    { return overlaps(ConstVectorView{static_cast<const double *>(pointer), count, view.device}, b); };
    return pointer_overlap(view.data, view.nnz) || pointer_overlap(view.indptr, view.rows + 1) ||
           pointer_overlap(view.indices, view.nnz);
}

} // namespace detail

} // namespace cuda
} // namespace cg
