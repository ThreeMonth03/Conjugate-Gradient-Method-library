#pragma once

#include <cg/backend/cuda/detail/runtime.hpp>
#include <cg/storage/cuda/device_memory.hpp>

#include <memory>

namespace cg
{
namespace cuda
{
namespace detail
{

struct PinnedDeleter
{
    void operator()(double *pointer) const noexcept;
};
using PinnedScalar = std::unique_ptr<double, PinnedDeleter>;
PinnedScalar pinned_scalar();

} // namespace detail
} // namespace cuda
} // namespace cg
