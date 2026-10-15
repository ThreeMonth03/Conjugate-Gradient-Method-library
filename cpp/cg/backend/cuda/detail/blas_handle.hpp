#pragma once

#include <cg/backend/cuda/detail/runtime.hpp>

#include <cublas_v2.h>

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
struct BlasHandle
{
    explicit BlasHandle(const Context &context) : device(context.device())
    {
        DeviceGuard guard(device);
        check(cublasCreate(&value), "cublasCreate");
        const auto status = cublasSetStream(value, stream(context));
        if (status != CUBLAS_STATUS_SUCCESS)
        {
            (void)cublasDestroy(value);
            value = nullptr;
            check(status, "cublasSetStream");
        }
    }
    BlasHandle(const BlasHandle &) = delete;
    BlasHandle &operator=(const BlasHandle &) = delete;
    ~BlasHandle()
    {
        on_device_noexcept(device, [&] { (void)cublasDestroy(value); });
    }
    int device;
    cublasHandle_t value = nullptr;
};

} // namespace detail
} // namespace cuda
} // namespace cg
