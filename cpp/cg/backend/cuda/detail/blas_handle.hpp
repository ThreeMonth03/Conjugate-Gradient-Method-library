#pragma once

#include <cg/runtime/cuda/detail/runtime.hpp>

#include <cublas_v2.h>

#include <memory>
#include <type_traits>

namespace cg
{
namespace cuda
{
namespace detail
{

inline void check(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
    }
}
struct BlasDeleter
{
    int device = -1;
    void operator()(cublasHandle_t handle) const noexcept
    {
        on_device_noexcept(device, [&] { (void)cublasDestroy(handle); });
    }
};
using BlasHandle = std::unique_ptr<std::remove_pointer_t<cublasHandle_t>, BlasDeleter>;

inline BlasHandle blas_handle(const Context &context)
{
    DeviceGuard guard(context.device());
    cublasHandle_t value = nullptr;
    check(cublasCreate(&value), "cublasCreate");
    BlasHandle handle(value, BlasDeleter{context.device()});
    check(cublasSetStream(handle.get(), stream(context)), "cublasSetStream");
    return handle;
}

} // namespace detail
} // namespace cuda
} // namespace cg
