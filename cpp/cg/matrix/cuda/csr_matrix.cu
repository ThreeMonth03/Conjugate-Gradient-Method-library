#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/detail/validation.hpp>
#include <cg/matrix/detail/csr_validation.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/runtime/detail/checked_size.hpp>

#include <limits>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cg
{
namespace cuda
{

namespace
{
void csr_dimensions(std::size_t rows, std::size_t cols, std::size_t nnz)
{
    const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
    if (rows > limit || cols > limit || nnz > limit)
    {
        throw std::invalid_argument("CUDA CSR dimensions exceed int64 range");
    }
}
template <class T>
std::vector<std::size_t> download_indices(const void *source, std::size_t size, const Context &context)
{
    std::vector<T> raw(size);
    if (size != 0)
    {
        detail::check(cudaMemcpyAsync(raw.data(), source, cg::detail::checked_product(size, sizeof(T)),
                                      cudaMemcpyDeviceToHost, detail::stream(context)),
                      "cudaMemcpyAsync CSR indices");
    }
    context.synchronize();
    std::vector<std::size_t> result;
    result.reserve(size);
    for (const auto value : raw)
    {
        if (value < 0)
        {
            throw std::invalid_argument("CUDA CSR indices must be nonnegative");
        }
        if (!std::in_range<std::size_t>(value))
        {
            throw std::invalid_argument("CUDA CSR indices exceed size_t range");
        }
        result.push_back(static_cast<std::size_t>(value));
    }
    return result;
}
void upload_indices(std::int64_t *destination, std::span<const std::size_t> source, const Context &context)
{
    const std::vector<std::int64_t> values(source.begin(), source.end());
    if (!values.empty())
    {
        detail::check(cudaMemcpyAsync(destination, values.data(), values.size() * sizeof(std::int64_t),
                                      cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpyAsync CSR indices upload");
    }
    context.synchronize();
}
} // namespace

CsrMatrix::CsrMatrix(const cg::CsrMatrix &host, const Context &context)
{
    csr_dimensions(host.rows(), host.cols(), host.nnz());
    m_rows = host.rows();
    m_cols = host.cols();
    m_indptr = detail::device_allocation<std::int64_t>(host.rows() + 1, context);
    m_indices = detail::device_allocation<std::int64_t>(host.nnz(), context);
    m_values = Vector(host.nnz(), context);
    detail::DeviceGuard guard(context.device());
    upload_indices(m_indptr.get(), host.indptr(), context);
    upload_indices(m_indices.get(), host.indices(), context);
    m_values.upload(host.values(), context);
}
CsrMatrix::CsrMatrix(CsrMatrixView source, const Context &context)
{
    csr_dimensions(source.rows(), source.cols(), source.nnz());
    if (source.index_type() != IndexType::int32 && source.index_type() != IndexType::int64)
    {
        throw std::invalid_argument("unsupported CUDA CSR index type");
    }
    const std::size_t width = source.index_type() == IndexType::int32 ? 4 : 8;
    detail::validate_pointer(source.indptr(), cg::detail::checked_product(source.rows() + 1, width), width,
                             source.device(), context);
    detail::validate_pointer(source.indices(), cg::detail::checked_product(source.nnz(), width), width, source.device(),
                             context);
    detail::validate(ConstVectorView{source.data(), source.nnz(), source.device()}, context);
    detail::DeviceGuard guard(context.device());
    const auto rowptr = source.index_type() == IndexType::int32
                            ? download_indices<std::int32_t>(source.indptr(), source.rows() + 1, context)
                            : download_indices<std::int64_t>(source.indptr(), source.rows() + 1, context);
    const auto columns = source.index_type() == IndexType::int32
                             ? download_indices<std::int32_t>(source.indices(), source.nnz(), context)
                             : download_indices<std::int64_t>(source.indices(), source.nnz(), context);
    cg::detail::validate_csr_structure(source.rows(), source.cols(), rowptr, columns, source.nnz());
    m_rows = source.rows();
    m_cols = source.cols();
    m_indptr = detail::device_allocation<std::int64_t>(source.rows() + 1, context);
    m_indices = detail::device_allocation<std::int64_t>(source.nnz(), context);
    m_values = Vector(source.nnz(), context);
    upload_indices(m_indptr.get(), rowptr, context);
    upload_indices(m_indices.get(), columns, context);
    m_values.copy_from_device({source.data(), source.nnz(), source.device()}, context);
}
CsrMatrix::CsrMatrix(CsrMatrix &&other) noexcept
    : m_rows(std::exchange(other.m_rows, 0)), m_cols(std::exchange(other.m_cols, 0)),
      m_indptr(std::move(other.m_indptr)), m_indices(std::move(other.m_indices)), m_values(std::move(other.m_values))
{
}
CsrMatrix &CsrMatrix::operator=(CsrMatrix &&other) noexcept
{
    if (this != &other)
    {
        m_rows = std::exchange(other.m_rows, 0);
        m_cols = std::exchange(other.m_cols, 0);
        m_indptr = std::move(other.m_indptr);
        m_indices = std::move(other.m_indices);
        m_values = std::move(other.m_values);
    }
    return *this;
}
std::size_t CsrMatrix::rows() const noexcept
{
    return m_rows;
}
std::size_t CsrMatrix::cols() const noexcept
{
    return m_cols;
}
std::size_t CsrMatrix::nnz() const noexcept
{
    return m_values.size();
}
int CsrMatrix::device() const noexcept
{
    return m_values.device();
}
CsrMatrixView CsrMatrix::view() const noexcept
{
    return {m_indptr.get(), m_indices.get(), m_values.data(), m_rows, m_cols, nnz(), IndexType::int64, device()};
}

} // namespace cuda
} // namespace cg
