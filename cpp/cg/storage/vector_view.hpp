#pragma once

#include <cstddef>

namespace cg
{

struct ConstVectorView
{
    const double *data = nullptr;
    std::size_t size = 0;
};

struct VectorView
{
    double *data = nullptr;
    std::size_t size = 0;

    operator ConstVectorView() const noexcept { return {data, size}; }
};

} // namespace cg
