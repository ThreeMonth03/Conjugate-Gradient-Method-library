#ifndef CG_CUDA_DETAIL_HPP
#define CG_CUDA_DETAIL_HPP

#include "cg/cuda.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <limits>
#include <stdexcept>
#include <string>

namespace cg
{
namespace cuda
{
namespace detail
{

inline void check(cudaError_t status, const char *operation)
{
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
inline void check(cublasStatus_t status, const char *operation)
{
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation) + ": cuBLAS status " +
                                 std::to_string(static_cast<int>(status)));
}
class DeviceGuard
{
  public:
    explicit DeviceGuard(int device)
    {
        check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != device)
            check(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceGuard() { (void)cudaSetDevice(previous_); }

  private:
    int previous_ = 0;
};
inline cudaStream_t stream(const Context &context)
{
    return reinterpret_cast<cudaStream_t>(context.stream());
}
inline std::size_t product(std::size_t a, std::size_t b)
{
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::invalid_argument("CUDA storage size overflows size_t");
    return a * b;
}
inline std::size_t span(DenseMatrixView A)
{
    if (A.row_stride < A.cols)
        throw std::invalid_argument("CUDA row stride is smaller than its columns");
    if (A.rows == 0 || A.cols == 0)
        return 0;
    const auto preceding = product(A.rows - 1, A.row_stride);
    if (preceding > std::numeric_limits<std::size_t>::max() - A.cols)
        throw std::invalid_argument("CUDA matrix span overflows size_t");
    return preceding + A.cols;
}
void validate(ConstVectorView view, const Context &context, bool inspect_pointer = true);
void validate_pointer(const void *data, std::size_t bytes, std::size_t alignment, int device,
                      const Context &context, bool inspect_pointer = true);
void validate(DenseMatrixView view, const Context &context, bool inspect_pointer = true);
bool overlaps(ConstVectorView a, ConstVectorView b);
bool overlaps(DenseMatrixView A, ConstVectorView b);
bool overlaps(const CsrMatrix &A, ConstVectorView b);
void fill(VectorView output, double value, const Context &context);
void initialize_jacobi(DenseMatrixView A, VectorView inverse, const Context &context);
void initialize_jacobi(CsrMatrixView A, VectorView inverse, const Context &context);

template <class T> class DeviceAllocation
{
  public:
    DeviceAllocation(std::size_t count, const Context &context) : device_(context.device())
    {
        const auto bytes = product(count, sizeof(T));
        DeviceGuard guard(device_);
        if (bytes != 0)
            check(cudaMalloc(reinterpret_cast<void **>(&data_), bytes), "cudaMalloc");
    }
    ~DeviceAllocation()
    {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device_);
        (void)cudaFree(data_);
        (void)cudaSetDevice(previous);
    }
    DeviceAllocation(const DeviceAllocation &) = delete;
    DeviceAllocation &operator=(const DeviceAllocation &) = delete;
    T *data() const noexcept { return data_; }

  private:
    int device_;
    T *data_ = nullptr;
};

struct PinnedScalar
{
    PinnedScalar()
    {
        check(cudaMallocHost(reinterpret_cast<void **>(&data), sizeof(double)), "cudaMallocHost");
    }
    ~PinnedScalar() { (void)cudaFreeHost(data); }
    double *data = nullptr;
};
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
    ~BlasHandle()
    {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device);
        (void)cublasDestroy(value);
        (void)cudaSetDevice(previous);
    }
    int device;
    cublasHandle_t value = nullptr;
};

constexpr unsigned int block_size = 256;
constexpr unsigned int max_blocks = 256;
inline unsigned int blocks(std::size_t size)
{
    if (size == 0)
        return 1;
    return static_cast<unsigned int>(
        std::min<std::size_t>((size - 1) / block_size + 1, max_blocks));
}

} // namespace detail

struct Workspace::Impl
{
    Impl(std::size_t size, const Context &ctx)
        : context(ctx), r(size, ctx), p(size, ctx), ap(size, ctx), scratch(size, ctx), z(size, ctx),
          partials(detail::max_blocks, ctx), scalar(1, ctx), blas_workspace(4 * 1024 * 1024, ctx),
          handle(ctx)
    {
        detail::DeviceGuard guard(ctx.device());
        detail::check(cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST),
                      "cublasSetPointerMode");
        detail::check(cublasSetAtomicsMode(handle.value, CUBLAS_ATOMICS_NOT_ALLOWED),
                      "cublasSetAtomicsMode");
        detail::check(cublasSetWorkspace(handle.value, blas_workspace.data(), 4 * 1024 * 1024),
                      "cublasSetWorkspace");
    }
    Context context;
    Vector r, p, ap, scratch, z;
    detail::DeviceAllocation<double> partials, scalar;
    detail::DeviceAllocation<unsigned char> blas_workspace;
    detail::PinnedScalar host;
    detail::BlasHandle handle;
    std::atomic<bool> active{false};
};

} // namespace cuda
} // namespace cg

#endif
