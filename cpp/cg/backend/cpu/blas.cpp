#include <cg/backend/cpu/backend.hpp>
#include <cg/backend/cpu/detail/blas.hpp>

#include <algorithm>
#include <limits>
#include <stdexcept>

#ifdef CG_ENABLE_BLAS
#include <cblas.h>

#ifndef CG_BLAS_PROVIDER_INFO
#define CG_BLAS_PROVIDER_INFO "OpenBLAS (sequential, LP64)"
#endif
#endif

namespace cg
{

bool CpuBackend::blas_enabled() noexcept
{
#ifdef CG_ENABLE_BLAS
    return true;
#else
    return false;
#endif
}

const char *CpuBackend::blas_info() noexcept
{
#ifdef CG_ENABLE_BLAS
    return CG_BLAS_PROVIDER_INFO;
#else
    return "disabled";
#endif
}

namespace detail
{

void blas_apply(DenseMatrixView A, ConstVectorView x, VectorView output)
{
#ifdef CG_ENABLE_BLAS
    static_assert(std::numeric_limits<int>::digits == 31 && std::numeric_limits<blasint>::digits == 31,
                  "the BLAS provider must use the signed int32 LP64 interface");
    const auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    if (A.rows > limit || A.cols > limit || A.row_stride > limit)
    {
        throw std::invalid_argument("dense matrix dimensions or row stride exceed BLAS int32 range");
    }
    if (A.rows == 0)
    {
        return;
    }
    if (A.cols == 0)
    {
        std::fill_n(output.data(), output.size(), 0.0);
        return;
    }
    cblas_dgemv(CblasRowMajor, CblasNoTrans, static_cast<int>(A.rows), static_cast<int>(A.cols), 1.0, A.data,
                static_cast<int>(A.row_stride), x.data(), 1, 0.0, output.data(), 1);
#else
    (void)A;
    (void)x;
    (void)output;
    throw std::invalid_argument("BLAS dense kernel requires a BLAS-enabled build");
#endif
}

} // namespace detail
} // namespace cg
