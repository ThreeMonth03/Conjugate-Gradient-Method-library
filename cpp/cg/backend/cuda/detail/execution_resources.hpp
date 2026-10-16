#pragma once

#include <cg/backend/cuda/dense_kernel.hpp>
#include <cg/backend/cuda/detail/blas_handle.hpp>
#include <cg/storage/cuda/device_memory.hpp>

#include <cstddef>
#include <optional>
#include <span>

namespace cg
{
namespace cuda
{
namespace detail
{

struct CublasResources
{
    CublasResources(const Context &context, std::span<std::byte> scratch);
    static constexpr std::size_t workspace_bytes = 4 * 1024 * 1024;
    // Borrows the full scratch prefix of ExecutionResources::storage.
    std::span<std::byte> workspace;
    BlasHandle handle;
};

// One execution's reduction and optional dense-provider storage. The backend
// borrows this state; it neither knows nor owns a solver workspace.
struct ExecutionResources
{
    explicit ExecutionResources(const Context &context, DenseKernel kernel = DenseKernel::native);
    ExecutionResources(ExecutionResources &&) = delete;
    ExecutionResources &operator=(ExecutionResources &&) = delete;
    Context context;
    // Declared before cublas so the handle is destroyed before its borrowed scratch.
    DeviceAllocation<std::byte> storage;
    // Borrowed nonoverlapping reduction spans, after any cuBLAS scratch.
    double *partials, *scalar;
    // Scalar copies are consumed only after the workspace stream is synchronized.
    double host = 0.0;
    std::optional<CublasResources> cublas;
    DenseKernel dense_kernel() const noexcept { return cublas ? DenseKernel::cublas : DenseKernel::native; }
};

} // namespace detail
} // namespace cuda
} // namespace cg
