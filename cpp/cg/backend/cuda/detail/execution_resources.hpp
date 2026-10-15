#pragma once

#include <cg/backend/cuda/dense_kernel.hpp>
#include <cg/backend/cuda/detail/blas_handle.hpp>
#include <cg/storage/cuda/detail/allocation.hpp>

#include <optional>

namespace cg
{
namespace cuda
{
namespace detail
{

struct CublasResources
{
    explicit CublasResources(const Context &context);
    static constexpr std::size_t workspace_bytes = 4 * 1024 * 1024;
    DeviceAllocation<unsigned char> workspace;
    BlasHandle handle;
};

// One execution's reduction and optional dense-provider storage. The backend
// borrows this state; it neither knows nor owns a solver workspace.
struct ExecutionResources
{
    explicit ExecutionResources(const Context &context, DenseKernel kernel = DenseKernel::native);
    Context context;
    DeviceAllocation<double> reduction;
    // Borrowed nonoverlapping spans of reduction, valid for this owner's lifetime.
    double *partials, *scalar;
    // Scalar copies are consumed only after the workspace stream is synchronized.
    double host = 0.0;
    std::optional<CublasResources> cublas;
    DenseKernel kernel;
};

} // namespace detail
} // namespace cuda
} // namespace cg
