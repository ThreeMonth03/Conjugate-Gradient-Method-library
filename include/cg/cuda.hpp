#ifndef CG_CUDA_HPP
#define CG_CUDA_HPP

#include "cg/csr.hpp"
#include "cg/solver.hpp"

#include <cstdint>
#include <memory>
#include <string>

namespace cg
{
namespace cuda
{

bool available() noexcept;
int device_count() noexcept;
int runtime_version();
int driver_version();

struct DeviceInfo
{
    std::string name;
    int major = 0;
    int minor = 0;
    std::size_t total_memory = 0;
};
DeviceInfo device_info(int device);

// The stream is borrowed, must belong to device, and must outlive every use.
// CUDA operations temporarily select this device on the calling thread and
// restore the previous selection. No process-wide stream setting is changed.
// Borrowed input data must be ready on this stream before submitting a solve.
class Context
{
  public:
    explicit Context(int device = 0, std::uintptr_t stream = 0);
    int device() const noexcept { return device_; }
    std::uintptr_t stream() const noexcept { return stream_; }
    void synchronize() const;

  private:
    int device_;
    std::uintptr_t stream_;
};

struct ConstVectorView
{
    const double *data = nullptr;
    std::size_t size = 0;
    int device = -1;
};
struct VectorView
{
    double *data = nullptr;
    std::size_t size = 0;
    int device = -1;
    operator ConstVectorView() const noexcept { return {data, size, device}; }
};
struct DenseMatrixView
{
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_stride = 0;
    int device = -1;
};
enum class IndexType
{
    int32,
    int64
};
struct CsrMatrixView
{
    const void *indptr = nullptr;
    const void *indices = nullptr;
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t nnz = 0;
    IndexType index_type = IndexType::int64;
    int device = -1;
};

class Vector
{
  public:
    Vector() = default;
    Vector(std::size_t size, const Context &context);
    Vector(cg::ConstVectorView host, const Context &context);
    ~Vector();
    Vector(const Vector &) = delete;
    Vector &operator=(const Vector &) = delete;
    Vector(Vector &&other) noexcept;
    Vector &operator=(Vector &&other) noexcept;
    std::size_t size() const noexcept { return size_; }
    int device() const noexcept { return device_; }
    double *data() noexcept { return data_; }
    const double *data() const noexcept { return data_; }
    VectorView view() noexcept { return {data_, size_, device_}; }
    ConstVectorView view() const noexcept { return {data_, size_, device_}; }
    void upload(cg::ConstVectorView host, const Context &context);
    void download(cg::VectorView host, const Context &context) const;
    void copy_from_device(ConstVectorView source, const Context &context);
    void fill(double value, const Context &context);

  private:
    double *data_ = nullptr;
    std::size_t size_ = 0;
    int device_ = -1;
};

class DenseMatrix
{
  public:
    DenseMatrix(cg::DenseMatrixView host, const Context &context);
    DenseMatrix(DenseMatrixView source, const Context &context);
    DenseMatrix(const DenseMatrix &) = delete;
    DenseMatrix &operator=(const DenseMatrix &) = delete;
    DenseMatrix(DenseMatrix &&other) noexcept;
    DenseMatrix &operator=(DenseMatrix &&other) noexcept;
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    int device() const noexcept { return values_.device(); }
    DenseMatrixView view() const noexcept
    {
        return {values_.data(), rows_, cols_, cols_, values_.device()};
    }

  private:
    std::size_t rows_;
    std::size_t cols_;
    Vector values_;
};

// Device snapshots canonicalize neither ordering nor duplicates: invalid CSR
// input is rejected after a one-time setup copy of its index arrays.
class CsrMatrix
{
  public:
    CsrMatrix(const cg::CsrMatrix &host, const Context &context);
    CsrMatrix(CsrMatrixView source, const Context &context);
    ~CsrMatrix();
    CsrMatrix(const CsrMatrix &) = delete;
    CsrMatrix &operator=(const CsrMatrix &) = delete;
    CsrMatrix(CsrMatrix &&) noexcept;
    CsrMatrix &operator=(CsrMatrix &&) noexcept;
    std::size_t rows() const noexcept;
    std::size_t cols() const noexcept;
    std::size_t nnz() const noexcept;
    int device() const noexcept;
    CsrMatrixView view() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

enum class DenseKernel
{
    native,
    cublas
};

class Backend;
// One active solve per workspace. The workspace owns its reduction buffers,
// pinned scalar and cuBLAS handle, all allocated before a prepared solve.
class Workspace
{
  public:
    Workspace(std::size_t size, const Context &context);
    ~Workspace();
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    std::size_t size() const noexcept;
    int device() const noexcept;
    std::uintptr_t stream() const noexcept;
    VectorView residual() noexcept;
    VectorView direction() noexcept;
    VectorView product() noexcept;
    VectorView scratch() noexcept;
    VectorView preconditioned() noexcept;
    bool try_acquire() noexcept;
    void release() noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class Backend;
};

class JacobiPreconditioner
{
  public:
    JacobiPreconditioner(DenseMatrixView A, const Context &context);
    JacobiPreconditioner(const CsrMatrix &A, const Context &context);
    std::size_t size() const noexcept { return inverse_.size(); }
    int device() const noexcept { return inverse_.device(); }
    ConstVectorView inverse_diagonal() const noexcept { return inverse_.view(); }
    double apply_dot(ConstVectorView r, VectorView z, const Backend &backend) const;

  private:
    Vector inverse_;
};

// Execution adapter for the shared solver engine. Public solve_into validates
// allocation/device attributes once; its kernels reuse the validated views.
class Backend
{
  public:
    Backend(const Context &context, Workspace &workspace, bool fused = true,
            DenseKernel kernel = DenseKernel::native);
    void copy(ConstVectorView source, VectorView destination) const;
    double dot(ConstVectorView x, ConstVectorView y) const;
    double norm(ConstVectorView x) const;
    double norm_from_squared(ConstVectorView x, double squared) const;
    double multiply_dot(ConstVectorView factors, ConstVectorView r, VectorView z) const;
    void axpy(double alpha, ConstVectorView x, VectorView y) const;
    void axpby(double alpha, ConstVectorView x, double beta, ConstVectorView y,
               VectorView output) const;
    void apply(DenseMatrixView A, ConstVectorView x, VectorView output) const;
    double apply_dot(DenseMatrixView A, ConstVectorView p, VectorView ap) const;
    double residual(DenseMatrixView A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    void apply(const CsrMatrix &A, ConstVectorView x, VectorView output) const;
    double apply_dot(const CsrMatrix &A, ConstVectorView p, VectorView ap) const;
    double residual(const CsrMatrix &A, ConstVectorView b, ConstVectorView x, VectorView r,
                    bool b_minus_ax = true) const;
    double update_x_r_norm2(VectorView x, VectorView r, ConstVectorView p, ConstVectorView ap,
                            double alpha) const;
    void update_direction(ConstVectorView r, VectorView p, double beta,
                          double residual_scale = 1.0) const;

  private:
    double finish_reduction(std::size_t blocks) const;
    Context context_;
    Workspace &workspace_;
    bool fused_;
    DenseKernel kernel_;
};

// Both algorithms and their numerical/refresh policy are shared with the CPU
// solver. Return is synchronous on context.stream(), including scalar checks.
SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const Context &context, const CgOptions &options = {}, bool fused = true,
                     DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const Context &context, const CgOptions &options = {}, bool fused = true,
                     DenseKernel kernel = DenseKernel::native,
                     const JacobiPreconditioner *M = nullptr);

} // namespace cuda
} // namespace cg

#endif
