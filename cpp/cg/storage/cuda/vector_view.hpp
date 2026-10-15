#pragma once

#include <cg/runtime/cuda/host_device.hpp>

#include <cstddef>

namespace cg
{
namespace cuda
{

// A borrowed device descriptor. Validation and synchronization belong to the
// consuming boundary; metadata access never inspects or dereferences storage.
class ConstVectorView
{
    public:
    CG_CUDA_HOST_DEVICE constexpr ConstVectorView() noexcept = default;
    CG_CUDA_HOST_DEVICE constexpr ConstVectorView(const double *data, std::size_t size, int device) noexcept
        : m_data(data), m_size(size), m_device(device)
    {
    }
    CG_CUDA_HOST_DEVICE constexpr const double *data() const noexcept { return m_data; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t size() const noexcept { return m_size; }
    CG_CUDA_HOST_DEVICE constexpr int device() const noexcept { return m_device; }

    private:
    const double *m_data = nullptr;
    std::size_t m_size = 0;
    int m_device = -1;
};
class VectorView
{
    public:
    CG_CUDA_HOST_DEVICE constexpr VectorView() noexcept = default;
    CG_CUDA_HOST_DEVICE constexpr VectorView(double *data, std::size_t size, int device) noexcept
        : m_data(data), m_size(size), m_device(device)
    {
    }
    CG_CUDA_HOST_DEVICE constexpr double *data() const noexcept { return m_data; }
    CG_CUDA_HOST_DEVICE constexpr std::size_t size() const noexcept { return m_size; }
    CG_CUDA_HOST_DEVICE constexpr int device() const noexcept { return m_device; }
    CG_CUDA_HOST_DEVICE constexpr operator ConstVectorView() const noexcept { return {m_data, m_size, m_device}; }

    private:
    double *m_data = nullptr;
    std::size_t m_size = 0;
    int m_device = -1;
};

} // namespace cuda
} // namespace cg
