#ifndef CG_STORAGE_HPP
#define CG_STORAGE_HPP

#include <cstddef>
#include <vector>

namespace cg
{

struct ConstVectorView
{
    const double *data = nullptr;
    std::size_t size = 0;
};

struct VectorView
{
    double *data = nullptr;
    std::size_t size = 0;

    operator ConstVectorView() const noexcept { return {data, size}; }
};

struct DenseMatrixView
{
    const double *data = nullptr;
    std::size_t rows = 0;
    std::size_t cols = 0;
    std::size_t row_stride = 0;

    DenseMatrixView() = default;
    DenseMatrixView(const double *data, std::size_t rows, std::size_t cols,
                    std::size_t row_stride = 0);
};

class Vector
{
  public:
    explicit Vector(std::size_t size = 0);
    explicit Vector(std::vector<double> values);
    std::size_t size() const noexcept { return values_.size(); }
    std::size_t capacity() const noexcept { return values_.capacity(); }
    double *data() noexcept { return values_.data(); }
    const double *data() const noexcept { return values_.data(); }
    VectorView view() noexcept { return {data(), size()}; }
    ConstVectorView view() const noexcept { return {data(), size()}; }
    void resize(std::size_t size);

  private:
    std::vector<double> values_;
};

class DenseMatrix
{
  public:
    DenseMatrix(std::size_t rows, std::size_t cols);
    DenseMatrix(std::size_t rows, std::size_t cols, std::vector<double> values);
    std::size_t rows() const noexcept { return rows_; }
    std::size_t cols() const noexcept { return cols_; }
    double *data() noexcept { return values_.data(); }
    const double *data() const noexcept { return values_.data(); }
    DenseMatrixView view() const { return {data(), rows_, cols_}; }

  private:
    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> values_;
};

// One workspace belongs to one active solve. Reuse it between calls; distinct
// workspaces are required for simultaneous solves, including on one backend.
class Workspace
{
  public:
    explicit Workspace(std::size_t size = 0);
    void prepare(std::size_t size);
    std::size_t size() const noexcept { return r_.size(); }
    std::size_t capacity() const noexcept { return r_.capacity(); }
    VectorView residual() noexcept { return r_.view(); }
    VectorView direction() noexcept { return p_.view(); }
    VectorView product() noexcept { return ap_.view(); }
    VectorView scratch() noexcept { return scratch_.view(); }

  private:
    Vector r_;
    Vector p_;
    Vector ap_;
    Vector scratch_;
};

namespace detail
{
void validate(ConstVectorView view);
void validate(DenseMatrixView view);
bool overlaps(ConstVectorView a, ConstVectorView b);
bool overlaps(DenseMatrixView a, ConstVectorView b);
} // namespace detail

} // namespace cg

#endif
