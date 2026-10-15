#include <cg/preconditioner/cuda/detail/kernels.hpp>
#include <cg/runtime/cuda/detail/launch.hpp>

namespace cg
{
namespace cuda
{
namespace
{

__global__ void dense_jacobi_kernel(DenseMatrixView A, VectorView inverse, int *invalid)
{
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < A.rows();
         i += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        const double diagonal = A.data()[i * A.row_stride() + i];
        const double value = 1.0 / diagonal;
        if (!isfinite(diagonal) || diagonal <= 0.0 || !isfinite(value) || value <= 0.0)
        {
            atomicExch(invalid, 1);
        }
        inverse.data()[i] = value;
    }
}

__global__ void csr_jacobi_kernel(CsrMatrixView A, VectorView inverse, int *invalid)
{
    const auto *indptr = static_cast<const std::int64_t *>(A.indptr());
    const auto *indices = static_cast<const std::int64_t *>(A.indices());
    for (std::size_t row = blockIdx.x * blockDim.x + threadIdx.x; row < A.rows();
         row += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    {
        double diagonal = 0.0;
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry)
        {
            if (indices[entry] == static_cast<std::int64_t>(row))
            {
                diagonal = A.data()[entry];
                break;
            }
        }
        const double value = 1.0 / diagonal;
        if (!isfinite(diagonal) || diagonal <= 0.0 || !isfinite(value) || value <= 0.0)
        {
            atomicExch(invalid, 1);
        }
        inverse.data()[row] = value;
    }
}

} // namespace

namespace detail
{
template <class Launch> void prepare_jacobi(const Context &context, int *invalid, Launch launch)
{
    DeviceGuard guard(context.device());
    check(cudaMemsetAsync(invalid, 0, sizeof(int), stream(context)), "cudaMemsetAsync Jacobi");
    launch(invalid);
    detail::launch_check();
    int host = 0;
    check(cudaMemcpyAsync(&host, invalid, sizeof(int), cudaMemcpyDeviceToHost, stream(context)),
          "cudaMemcpyAsync Jacobi validation");
    context.synchronize();
    if (host != 0)
    {
        throw std::invalid_argument("CUDA Jacobi requires a finite positive diagonal and inverse");
    }
}
void initialize_jacobi(DenseMatrixView A, VectorView inverse, int *invalid, const Context &context)
{
    prepare_jacobi(context, invalid, [&](int *flag)
                   { dense_jacobi_kernel<<<blocks(A.rows()), block_size, 0, stream(context)>>>(A, inverse, flag); });
}
void initialize_jacobi(CsrMatrixView A, VectorView inverse, int *invalid, const Context &context)
{
    prepare_jacobi(context, invalid, [&](int *flag)
                   { csr_jacobi_kernel<<<blocks(A.rows()), block_size, 0, stream(context)>>>(A, inverse, flag); });
}
} // namespace detail

} // namespace cuda
} // namespace cg
