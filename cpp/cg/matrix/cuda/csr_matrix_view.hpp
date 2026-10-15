#pragma once

#include <cg/runtime/cuda/host_device.hpp>

#include <cstddef>

namespace cg
{
namespace cuda
{

enum class IndexType
{
    int32,
    int64
};
// A borrowed CSR descriptor. Index width remains explicit for int32/int64
// interoperability; the upload boundary validates and normalizes the indices.
class CsrMatrixView
{
    public:
    CG_CUDA_HOST_DEVICE constexpr CsrMatrixView() noexcept = default;
    CG_CUDA_HOST_DEVICE constexpr CsrMatrixView(const void *indptr, const void *indices, const double *data,
                                                std::size_t rows, std::size_t cols, std::size_t nnz,
                                                IndexType index_type, int device) noexcept
        : m_indptr(indptr), m_indices(indices), m_data(data), m_rows(rows), m_cols(cols), m_nnz(nnz),
          m_index_type(index_type), m_device(device)
    {
    }
    CG_CUDA_HOST_DEVICE constexpr const void *indptr() const noexcept { return m_indptr; }
    CG_CUDA_HOST_DEVICE constexpr const void *indices() const noexcept { return m_indices; }
    CG_CUDA_HOST_DEVICE constexpr const double *data() const noexcept { return m_data; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t rows() const noexcept { return m_rows; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t cols() const noexcept { return m_cols; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t nnz() const noexcept { return m_nnz; }
    CG_CUDA_HOST_DEVICE constexpr IndexType index_type() const noexcept { return m_index_type; }
    CG_CUDA_HOST_DEVICE constexpr int device() const noexcept { return m_device; }

    private:
    const void *m_indptr = nullptr;
    const void *m_indices = nullptr;
    const double *m_data = nullptr;
    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    std::size_t m_nnz = 0;
    IndexType m_index_type = IndexType::int64;
    int m_device = -1;
};

} // namespace cuda
} // namespace cg
