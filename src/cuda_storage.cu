#include "cg/detail/cuda.hpp"

#include <utility>
#include <vector>

namespace cg
{
namespace cuda
{

int device_count() noexcept
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess)
    {
        (void)cudaGetLastError();
        return 0;
    }
    return count;
}
bool available() noexcept { return device_count() > 0; }
int runtime_version()
{
    int version = 0;
    detail::check(cudaRuntimeGetVersion(&version), "cudaRuntimeGetVersion");
    return version;
}
int driver_version()
{
    int version = 0;
    detail::check(cudaDriverGetVersion(&version), "cudaDriverGetVersion");
    return version;
}
DeviceInfo device_info(int device)
{
    cudaDeviceProp properties{};
    detail::check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    return {properties.name, properties.major, properties.minor, properties.totalGlobalMem};
}

Context::Context(int device, std::uintptr_t stream) : device_(device), stream_(stream)
{
    int count = 0;
    detail::check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (device < 0 || device >= count)
        throw std::invalid_argument("CUDA device is out of range");
    detail::DeviceGuard guard(device);
    int stream_device = -1;
    detail::check(cudaStreamGetDevice(detail::stream(*this), &stream_device),
                  "cudaStreamGetDevice");
    if (stream_device != device)
        throw std::invalid_argument("CUDA stream belongs to a different device");
    unsigned int flags = 0;
    detail::check(cudaStreamGetFlags(detail::stream(*this), &flags), "cudaStreamGetFlags");
}
void Context::synchronize() const
{
    detail::DeviceGuard guard(device_);
    detail::check(cudaStreamSynchronize(detail::stream(*this)), "cudaStreamSynchronize");
}

namespace detail
{

void validate_pointer(const void *data, std::size_t bytes, std::size_t alignment, int device,
                      const Context &context, bool inspect_pointer)
{
    if (bytes == 0)
        return;
    if (device != context.device())
        throw std::invalid_argument("CUDA storage belongs to a different device");
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || begin % alignment != 0 ||
        begin > std::numeric_limits<std::uintptr_t>::max() - bytes)
        throw std::invalid_argument("invalid CUDA storage pointer, alignment or span");
    if (inspect_pointer)
    {
        DeviceGuard guard(device);
        cudaPointerAttributes attributes{};
        const auto status = cudaPointerGetAttributes(&attributes, data);
        if (status != cudaSuccess)
        {
            (void)cudaGetLastError();
            throw std::invalid_argument("expected CUDA device storage");
        }
        if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
            throw std::invalid_argument("expected storage on the selected CUDA device");
    }
}
void validate(ConstVectorView view, const Context &context, bool inspect_pointer)
{
    validate_pointer(view.data, product(view.size, sizeof(double)), alignof(double), view.device,
                     context, inspect_pointer);
}
void validate(DenseMatrixView view, const Context &context, bool inspect_pointer)
{
    validate(ConstVectorView{view.data, span(view), view.device}, context, inspect_pointer);
}
bool overlaps(ConstVectorView a, ConstVectorView b)
{
    if (a.size == 0 || b.size == 0 || a.device != b.device)
        return false;
    const auto abegin = reinterpret_cast<std::uintptr_t>(a.data);
    const auto bbegin = reinterpret_cast<std::uintptr_t>(b.data);
    const auto abytes = product(a.size, sizeof(double)), bbytes = product(b.size, sizeof(double));
    if (abegin > std::numeric_limits<std::uintptr_t>::max() - abytes ||
        bbegin > std::numeric_limits<std::uintptr_t>::max() - bbytes)
        throw std::invalid_argument("CUDA storage address span overflows uintptr_t");
    return abegin < bbegin + bbytes && bbegin < abegin + abytes;
}
bool overlaps(DenseMatrixView A, ConstVectorView b)
{
    return overlaps(ConstVectorView{A.data, span(A), A.device}, b);
}
bool overlaps(const CsrMatrix &A, ConstVectorView b)
{
    const auto view = A.view();
    const auto pointer_overlap = [&](const void *pointer, std::size_t count)
    {
        return overlaps(ConstVectorView{static_cast<const double *>(pointer), count, view.device},
                        b);
    };
    return pointer_overlap(view.data, view.nnz) || pointer_overlap(view.indptr, view.rows + 1) ||
           pointer_overlap(view.indices, view.nnz);
}

} // namespace detail

Vector::Vector(std::size_t size, const Context &context) : size_(size), device_(context.device())
{
    const auto bytes = detail::product(size, sizeof(double));
    detail::DeviceGuard guard(device_);
    if (bytes != 0)
        detail::check(cudaMalloc(reinterpret_cast<void **>(&data_), bytes), "cudaMalloc");
}
Vector::Vector(cg::ConstVectorView host, const Context &context) : Vector(host.size, context)
{
    upload(host, context);
}
Vector::~Vector()
{
    if (data_ != nullptr)
    {
        int previous = 0;
        (void)cudaGetDevice(&previous);
        (void)cudaSetDevice(device_);
        (void)cudaFree(data_);
        (void)cudaSetDevice(previous);
    }
}
Vector::Vector(Vector &&other) noexcept
    : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)),
      device_(std::exchange(other.device_, -1))
{
}
Vector &Vector::operator=(Vector &&other) noexcept
{
    if (this != &other)
    {
        Vector temporary(std::move(other));
        std::swap(data_, temporary.data_);
        std::swap(size_, temporary.size_);
        std::swap(device_, temporary.device_);
    }
    return *this;
}
void Vector::upload(cg::ConstVectorView host, const Context &context)
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size != size_)
        throw std::invalid_argument("CUDA upload dimensions do not match");
    detail::DeviceGuard guard(context.device());
    if (size_ != 0)
        detail::check(cudaMemcpyAsync(data_, host.data, detail::product(size_, sizeof(double)),
                                      cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpyAsync upload");
    context.synchronize();
}
void Vector::download(cg::VectorView host, const Context &context) const
{
    cg::detail::validate(host);
    detail::validate(view(), context);
    if (host.size != size_)
        throw std::invalid_argument("CUDA download dimensions do not match");
    detail::DeviceGuard guard(context.device());
    if (size_ != 0)
        detail::check(cudaMemcpyAsync(host.data, data_, detail::product(size_, sizeof(double)),
                                      cudaMemcpyDeviceToHost, detail::stream(context)),
                      "cudaMemcpyAsync download");
    context.synchronize();
}
void Vector::copy_from_device(ConstVectorView source, const Context &context)
{
    detail::validate(source, context);
    detail::validate(view(), context);
    if (source.size != size_ || (source.data != data_ && detail::overlaps(source, view())))
        throw std::invalid_argument("CUDA device copy dimensions or storage overlap");
    detail::DeviceGuard guard(context.device());
    if (size_ != 0 && source.data != data_)
        detail::check(cudaMemcpyAsync(data_, source.data, detail::product(size_, sizeof(double)),
                                      cudaMemcpyDeviceToDevice, detail::stream(context)),
                      "cudaMemcpyAsync device copy");
    context.synchronize();
}
void Vector::fill(double value, const Context &context)
{
    detail::validate(view(), context);
    detail::fill(view(), value, context);
    context.synchronize();
}

DenseMatrix::DenseMatrix(cg::DenseMatrixView host, const Context &context)
    : rows_(host.rows), cols_(host.cols), values_(detail::product(host.rows, host.cols), context)
{
    cg::detail::validate(host);
    detail::DeviceGuard guard(context.device());
    if (rows_ != 0 && cols_ != 0)
        detail::check(cudaMemcpy2DAsync(values_.data(), cols_ * sizeof(double), host.data,
                                        host.row_stride * sizeof(double), cols_ * sizeof(double),
                                        rows_, cudaMemcpyHostToDevice, detail::stream(context)),
                      "cudaMemcpy2DAsync matrix upload");
    context.synchronize();
}
DenseMatrix::DenseMatrix(DenseMatrixView source, const Context &context)
    : rows_(source.rows), cols_(source.cols),
      values_(detail::product(source.rows, source.cols), context)
{
    detail::validate(source, context);
    detail::DeviceGuard guard(context.device());
    if (rows_ != 0 && cols_ != 0)
        detail::check(cudaMemcpy2DAsync(values_.data(), cols_ * sizeof(double), source.data,
                                        source.row_stride * sizeof(double), cols_ * sizeof(double),
                                        rows_, cudaMemcpyDeviceToDevice, detail::stream(context)),
                      "cudaMemcpy2DAsync matrix snapshot");
    context.synchronize();
}
DenseMatrix::DenseMatrix(DenseMatrix &&other) noexcept
    : rows_(std::exchange(other.rows_, 0)), cols_(std::exchange(other.cols_, 0)),
      values_(std::move(other.values_))
{
}
DenseMatrix &DenseMatrix::operator=(DenseMatrix &&other) noexcept
{
    if (this != &other)
    {
        rows_ = std::exchange(other.rows_, 0);
        cols_ = std::exchange(other.cols_, 0);
        values_ = std::move(other.values_);
    }
    return *this;
}

struct CsrMatrix::Impl
{
    Impl(std::size_t r, std::size_t c, std::size_t n, const Context &context)
        : rows(r), cols(c), nnz(n), device(context.device()), indptr(r + 1, context),
          indices(n, context), values(n, context)
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
        throw std::invalid_argument("CUDA CSR dimensions exceed int64 range");
}
template <class T>
std::vector<std::size_t> download_indices(const void *source, std::size_t size,
                                          const Context &context)
{
    std::vector<T> raw(size);
    if (size != 0)
        detail::check(cudaMemcpyAsync(raw.data(), source, detail::product(size, sizeof(T)),
                                      cudaMemcpyDeviceToHost, detail::stream(context)),
                      "cudaMemcpyAsync CSR indices");
    context.synchronize();
    std::vector<std::size_t> result;
    result.reserve(size);
    for (const auto value : raw)
    {
        if (value < 0)
            throw std::invalid_argument("CUDA CSR indices must be nonnegative");
        result.push_back(static_cast<std::size_t>(value));
    }
    return result;
}
void upload_indices(std::int64_t *destination, const std::vector<std::size_t> &source,
                    const Context &context)
{
    const std::vector<std::int64_t> values(source.begin(), source.end());
    if (!values.empty())
        detail::check(cudaMemcpyAsync(destination, values.data(),
                                      values.size() * sizeof(std::int64_t), cudaMemcpyHostToDevice,
                                      detail::stream(context)),
                      "cudaMemcpyAsync CSR indices upload");
    context.synchronize();
}
} // namespace

CsrMatrix::CsrMatrix(const cg::CsrMatrix &host, const Context &context)
{
    csr_dimensions(host.rows(), host.cols(), host.nnz());
    impl_ = std::make_unique<Impl>(host.rows(), host.cols(), host.nnz(), context);
    detail::DeviceGuard guard(context.device());
    upload_indices(impl_->indptr.data(), host.indptr(), context);
    upload_indices(impl_->indices.data(), host.indices(), context);
    impl_->values.upload({host.values().data(), host.nnz()}, context);
}
CsrMatrix::CsrMatrix(CsrMatrixView source, const Context &context)
{
    csr_dimensions(source.rows, source.cols, source.nnz);
    if (source.index_type != IndexType::int32 && source.index_type != IndexType::int64)
        throw std::invalid_argument("unsupported CUDA CSR index type");
    const std::size_t width = source.index_type == IndexType::int32 ? 4 : 8;
    detail::validate_pointer(source.indptr, detail::product(source.rows + 1, width), width,
                             source.device, context);
    detail::validate_pointer(source.indices, detail::product(source.nnz, width), width,
                             source.device, context);
    detail::validate(ConstVectorView{source.data, source.nnz, source.device}, context);
    detail::DeviceGuard guard(context.device());
    const auto rowptr =
        source.index_type == IndexType::int32
            ? download_indices<std::int32_t>(source.indptr, source.rows + 1, context)
            : download_indices<std::int64_t>(source.indptr, source.rows + 1, context);
    const auto columns = source.index_type == IndexType::int32
                             ? download_indices<std::int32_t>(source.indices, source.nnz, context)
                             : download_indices<std::int64_t>(source.indices, source.nnz, context);
    const cg::CsrMatrix validated(source.rows, source.cols, rowptr, columns,
                                  std::vector<double>(source.nnz));
    impl_ = std::make_unique<Impl>(source.rows, source.cols, source.nnz, context);
    upload_indices(impl_->indptr.data(), validated.indptr(), context);
    upload_indices(impl_->indices.data(), validated.indices(), context);
    impl_->values.copy_from_device({source.data, source.nnz, source.device}, context);
}
CsrMatrix::~CsrMatrix() = default;
CsrMatrix::CsrMatrix(CsrMatrix &&) noexcept = default;
CsrMatrix &CsrMatrix::operator=(CsrMatrix &&) noexcept = default;
std::size_t CsrMatrix::rows() const noexcept { return impl_ ? impl_->rows : 0; }
std::size_t CsrMatrix::cols() const noexcept { return impl_ ? impl_->cols : 0; }
std::size_t CsrMatrix::nnz() const noexcept { return impl_ ? impl_->nnz : 0; }
int CsrMatrix::device() const noexcept { return impl_ ? impl_->device : -1; }
CsrMatrixView CsrMatrix::view() const noexcept
{
    if (!impl_)
        return {};
    return {impl_->indptr.data(), impl_->indices.data(), impl_->values.data(),
            impl_->rows,          impl_->cols,           impl_->nnz,
            IndexType::int64,     impl_->device};
}

Workspace::Workspace(std::size_t size, const Context &context)
    : impl_(std::make_unique<Impl>(size, context))
{
}
Workspace::~Workspace() = default;
std::size_t Workspace::size() const noexcept { return impl_->r.size(); }
int Workspace::device() const noexcept { return impl_->context.device(); }
std::uintptr_t Workspace::stream() const noexcept { return impl_->context.stream(); }
VectorView Workspace::residual() noexcept { return impl_->r.view(); }
VectorView Workspace::direction() noexcept { return impl_->p.view(); }
VectorView Workspace::product() noexcept { return impl_->ap.view(); }
VectorView Workspace::scratch() noexcept { return impl_->scratch.view(); }
VectorView Workspace::preconditioned() noexcept { return impl_->z.view(); }
bool Workspace::try_acquire() noexcept { return !impl_->active.exchange(true); }
void Workspace::release() noexcept { impl_->active.store(false); }

JacobiPreconditioner::JacobiPreconditioner(DenseMatrixView A, const Context &context)
    : inverse_(A.rows, context)
{
    detail::validate(A, context);
    if (A.rows == 0 || A.rows != A.cols)
        throw std::invalid_argument("CUDA Jacobi requires a nonempty square matrix");
    detail::initialize_jacobi(A, inverse_.view(), context);
}
JacobiPreconditioner::JacobiPreconditioner(const CsrMatrix &A, const Context &context)
    : inverse_(A.rows(), context)
{
    if (A.device() != context.device() || A.rows() == 0 || A.rows() != A.cols())
        throw std::invalid_argument("CUDA Jacobi requires a square matrix on the selected device");
    detail::initialize_jacobi(A.view(), inverse_.view(), context);
}
double JacobiPreconditioner::apply_dot(ConstVectorView r, VectorView z,
                                       const Backend &backend) const
{
    return backend.multiply_dot(inverse_.view(), r, z);
}

} // namespace cuda
} // namespace cg
