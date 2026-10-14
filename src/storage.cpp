#include "cg/storage.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cg
{
namespace
{

std::size_t checked_product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
    {
        throw std::invalid_argument("storage size overflows size_t");
    }
    return a * b;
}

std::size_t matrix_span(DenseMatrixView view)
{
    if (view.rows == 0 || view.cols == 0)
    {
        return 0;
    }
    const auto preceding = checked_product(view.rows - 1, view.row_stride);
    if (preceding > std::numeric_limits<std::size_t>::max() - view.cols)
    {
        throw std::invalid_argument("matrix span overflows size_t");
    }
    return preceding + view.cols;
}

std::uintptr_t end_address(const double *data, std::size_t size)
{
    const auto bytes = checked_product(size, sizeof(double));
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
    {
        throw std::invalid_argument("storage address range overflows uintptr_t");
    }
    return begin + bytes;
}

} // namespace

DenseMatrixView::DenseMatrixView(const double *data, std::size_t rows, std::size_t cols,
                                 std::size_t row_stride)
    : data(data), rows(rows), cols(cols), row_stride(row_stride == 0 ? cols : row_stride)
{
    detail::validate(*this);
}

Vector::Vector(std::size_t size) : values_(size) {}
Vector::Vector(std::vector<double> values) : values_(std::move(values)) {}
void Vector::resize(std::size_t size) { values_.resize(size); }

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols)
    : rows_(rows), cols_(cols), values_(checked_product(rows, cols))
{
}

DenseMatrix::DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values)
    : rows_(rows), cols_(cols), values_(std::move(values))
{
    if (values_.size() != checked_product(rows, cols))
    {
        throw std::invalid_argument("matrix shape does not match storage size");
    }
}

Workspace::Workspace(std::size_t size) : r_(size), p_(size), ap_(size), scratch_(size) {}

void Workspace::prepare(std::size_t size)
{
    if (size > capacity())
    {
        *this = Workspace(size);
        return;
    }
    r_.resize(size);
    p_.resize(size);
    ap_.resize(size);
    scratch_.resize(size);
}

namespace detail
{

void validate(ConstVectorView view)
{
    if (view.size != 0 && view.data == nullptr)
    {
        throw std::invalid_argument("nonempty vector has a null data pointer");
    }
    if (view.size != 0 && reinterpret_cast<std::uintptr_t>(view.data) % alignof(double) != 0)
    {
        throw std::invalid_argument("float64 storage must be aligned for double access");
    }
    end_address(view.data, view.size);
}

void validate(DenseMatrixView view)
{
    if (view.row_stride < view.cols)
    {
        throw std::invalid_argument("matrix row stride is smaller than its columns");
    }
    validate(ConstVectorView{view.data, matrix_span(view)});
}

bool overlaps(ConstVectorView a, ConstVectorView b)
{
    validate(a);
    validate(b);
    if (a.size == 0 || b.size == 0)
    {
        return false;
    }
    return reinterpret_cast<std::uintptr_t>(a.data) < end_address(b.data, b.size) &&
           reinterpret_cast<std::uintptr_t>(b.data) < end_address(a.data, a.size);
}

bool overlaps(DenseMatrixView a, ConstVectorView b)
{
    validate(a);
    return overlaps(ConstVectorView{a.data, matrix_span(a)}, b);
}

} // namespace detail
} // namespace cg
