#include <cg/backend/cpu/backend.hpp>

#include <stdexcept>

namespace cg
{

CpuBackend::CpuBackend(int num_threads, bool fused, DenseKernel dense_kernel)
    : m_num_threads(num_threads), m_fused(fused), m_dense_kernel(dense_kernel)
{
    if (num_threads < 1)
    {
        throw std::invalid_argument("num_threads must be a positive integer");
    }
    if (dense_kernel != DenseKernel::native && dense_kernel != DenseKernel::blas)
    {
        throw std::invalid_argument("unsupported dense kernel");
    }
    if (dense_kernel == DenseKernel::blas)
    {
        if (!blas_enabled())
        {
            throw std::invalid_argument("BLAS dense kernel requires a BLAS-enabled build");
        }
        if (num_threads != 1)
        {
            throw std::invalid_argument("BLAS dense kernel requires num_threads=1");
        }
    }
    else if (num_threads > 1 && !openmp_enabled())
    {
        throw std::invalid_argument("multiple threads require an OpenMP build");
    }
}

bool CpuBackend::openmp_enabled() noexcept
{
#ifdef _OPENMP
    return true;
#else
    return false;
#endif
}

} // namespace cg
