#include <cg/matrix/cuda/dense_matrix.hpp>
#include <cg/matrix/cuda/detail/validation.hpp>
#include <cg/matrix/detail/validation.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <utility>

namespace cg
{
namespace cuda
{

DenseMatrix::DenseMatrix(cg::DenseMatrixView host, const Context &context)
    : m_rows(host.rows()), m_cols(host.cols()), m_values(cg::detail::checked_product(host.rows(), host.cols()), context)
{
    cg::detail::validate(host);
    detail::DeviceGuard guard(context.device());
    if (m_rows != 0 && m_cols != 0)
    {
        detail::check(cudaMemcpy2DAsync(m_values.data(), m_cols * sizeof(double), host.data(),
                                        host.row_stride() * sizeof(double), m_cols * sizeof(double), m_rows,
                                        cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpy2DAsync matrix upload");
    }
    context.synchronize();
}
DenseMatrix::DenseMatrix(DenseMatrixView source, const Context &context)
    : m_rows(source.rows()), m_cols(source.cols()),
      m_values(cg::detail::checked_product(source.rows(), source.cols()), context)
{
    detail::validate(source, context);
    detail::DeviceGuard guard(context.device());
    if (m_rows != 0 && m_cols != 0)
    {
        detail::check(cudaMemcpy2DAsync(m_values.data(), m_cols * sizeof(double), source.data(),
                                        source.row_stride() * sizeof(double), m_cols * sizeof(double), m_rows,
                                        cudaMemcpyDeviceToDevice, detail::stream(context)),
                      "cudaMemcpy2DAsync matrix snapshot");
    }
    context.synchronize();
}
DenseMatrix::DenseMatrix(DenseMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_values(std::move(other.m_values))
{
}
DenseMatrix &DenseMatrix::operator=(DenseMatrix &&other) noexcept
{
    if (this != &other)
    {
        m_rows = std::exchange(other.m_rows, 0);
        m_cols = std::exchange(other.m_cols, 0);
        m_values = std::move(other.m_values);
    }
    return *this;
}

} // namespace cuda
} // namespace cg
