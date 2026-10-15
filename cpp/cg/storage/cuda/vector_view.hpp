#pragma once

#include <cstddef>

namespace cg
{
namespace cuda
{

struct ConstVectorView
{
    const double *data = nullptr;
    std::size_t size = 0;
    int device = -1;
};
struct VectorView
{
    double *data = nullptr;
    std::size_t size = 0;
    int device = -1;
    operator ConstVectorView() const noexcept { return {data, size, device}; }
};

} // namespace cuda
} // namespace cg
