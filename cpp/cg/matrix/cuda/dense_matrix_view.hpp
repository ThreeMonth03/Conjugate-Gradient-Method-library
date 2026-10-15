#pragma once

#include <cg/runtime/cuda/host_device.hpp>

#include <cstddef>

namespace cg
{
namespace cuda
{

// Nonowning device metadata; constructing a view performs no validation.
class DenseMatrixView
{
    public:
    CG_CUDA_HOST_DEVICE constexpr DenseMatrixView() noexcept = default;
    CG_CUDA_HOST_DEVICE constexpr DenseMatrixView(const double *data, std::size_t rows, std::size_t cols,
                                                  std::size_t row_stride, int device) noexcept
        : m_data(data), m_rows(rows), m_cols(cols), m_row_stride(row_stride), m_device(device)
    {
    }
    CG_CUDA_HOST_DEVICE constexpr const double *data() const noexcept { return m_data; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t rows() const noexcept { return m_rows; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t cols() const noexcept { return m_cols; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t row_stride() const noexcept { return m_row_stride; }
    CG_CUDA_HOST_DEVICE constexpr int device() const noexcept { return m_device; }

    private:
    const double *m_data = nullptr;
    std::size_t m_rows = 0;
    std::size_t m_cols = 0;
    std::size_t m_row_stride = 0;
    int m_device = -1;
};

} // namespace cuda
} // namespace cg
