#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/validation.hpp>
#include <cg/storage/cuda/detail/allocation.hpp>
#include <cg/storage/cuda/vector.hpp>

#include <utility>
#include <vector>

namespace cg
{
namespace cuda
{

struct CsrMatrix::Impl
{
    Impl(std::size_t r, std::size_t c, std::size_t n, const Context &context)
        : rows(r), cols(c), nnz(n), device(context.device()), indptr(r + 1, context), indices(n, context),
          values(n, context)
    {
    }
    std::size_t rows, cols, nnz;
    int device;
    detail::DeviceAllocation<std::int64_t> indptr, indices;
    Vector values;
};

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
        detail::check(cudaMemcpyAsync(raw.data(), source, detail::product(size, sizeof(T)), cudaMemcpyDeviceToHost,
                                      detail::stream(context)),
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
        result.push_back(static_cast<std::size_t>(value));
    }
    return result;
}
void upload_indices(std::int64_t *destination, const std::vector<std::size_t> &source, const Context &context)
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
    m_impl = std::make_unique<Impl>(host.rows(), host.cols(), host.nnz(), context);
    detail::DeviceGuard guard(context.device());
    upload_indices(m_impl->indptr.data(), host.indptr(), context);
    upload_indices(m_impl->indices.data(), host.indices(), context);
    m_impl->values.upload({host.values().data(), host.nnz()}, context);
}
CsrMatrix::CsrMatrix(CsrMatrixView source, const Context &context)
{
    csr_dimensions(source.rows, source.cols, source.nnz);
    if (source.index_type != IndexType::int32 && source.index_type != IndexType::int64)
    {
        throw std::invalid_argument("unsupported CUDA CSR index type");
    }
    const std::size_t width = source.index_type == IndexType::int32 ? 4 : 8;
    detail::validate_pointer(source.indptr, detail::product(source.rows + 1, width), width, source.device, context);
    detail::validate_pointer(source.indices, detail::product(source.nnz, width), width, source.device, context);
    detail::validate(ConstVectorView{source.data, source.nnz, source.device}, context);
    detail::DeviceGuard guard(context.device());
    const auto rowptr = source.index_type == IndexType::int32
                            ? download_indices<std::int32_t>(source.indptr, source.rows + 1, context)
                            : download_indices<std::int64_t>(source.indptr, source.rows + 1, context);
    const auto columns = source.index_type == IndexType::int32
                             ? download_indices<std::int32_t>(source.indices, source.nnz, context)
                             : download_indices<std::int64_t>(source.indices, source.nnz, context);
    const cg::CsrMatrix validated(source.rows, source.cols, rowptr, columns, std::vector<double>(source.nnz));
    m_impl = std::make_unique<Impl>(source.rows, source.cols, source.nnz, context);
    upload_indices(m_impl->indptr.data(), validated.indptr(), context);
    upload_indices(m_impl->indices.data(), validated.indices(), context);
    m_impl->values.copy_from_device({source.data, source.nnz, source.device}, context);
}
CsrMatrix::~CsrMatrix() = default;
CsrMatrix::CsrMatrix(CsrMatrix &&) noexcept = default;
CsrMatrix &CsrMatrix::operator=(CsrMatrix &&) noexcept = default;
std::size_t CsrMatrix::rows() const noexcept
{
    return m_impl ? m_impl->rows : 0;
}
std::size_t CsrMatrix::cols() const noexcept
{
    return m_impl ? m_impl->cols : 0;
}
std::size_t CsrMatrix::nnz() const noexcept
{
    return m_impl ? m_impl->nnz : 0;
}
int CsrMatrix::device() const noexcept
{
    return m_impl ? m_impl->device : -1;
}
CsrMatrixView CsrMatrix::view() const noexcept
{
    if (!m_impl)
    {
        return {};
    }
    return {m_impl->indptr.data(), m_impl->indices.data(), m_impl->values.data(), m_impl->rows, m_impl->cols,
            m_impl->nnz,           IndexType::int64,       m_impl->device};
}

} // namespace cuda
} // namespace cg
