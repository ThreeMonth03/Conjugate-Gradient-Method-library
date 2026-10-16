#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/csr_matrix_view.hpp>
#include <cg/matrix/cuda/dense_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix_view.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/solver/cuda/workspace.hpp>
#include <cg/storage/cuda/vector.hpp>
#include <cg/storage/cuda/vector_view.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <type_traits>
#include <utility>

namespace
{
namespace gpu = cg::cuda;

template <class Owner> constexpr bool has_data = requires(Owner &&owner) { std::forward<Owner>(owner).data(); };
template <class Owner> constexpr bool has_view = requires(Owner &&owner) { std::forward<Owner>(owner).view(); };
template <class Owner>
constexpr bool has_inverse_diagonal = requires(Owner &&owner) { std::forward<Owner>(owner).inverse_diagonal(); };
template <class Owner> constexpr bool has_context = requires(Owner &&owner) { std::forward<Owner>(owner).context(); };
template <class Owner> constexpr bool has_residual = requires(Owner &&owner) { std::forward<Owner>(owner).residual(); };
template <class Owner>
constexpr bool has_direction = requires(Owner &&owner) { std::forward<Owner>(owner).direction(); };
template <class Owner> constexpr bool has_product = requires(Owner &&owner) { std::forward<Owner>(owner).product(); };
template <class Owner> constexpr bool has_scratch = requires(Owner &&owner) { std::forward<Owner>(owner).scratch(); };
template <class Owner>
constexpr bool has_preconditioned = requires(Owner &&owner) { std::forward<Owner>(owner).preconditioned(); };

// Owner lifetime restrictions are testable without a CUDA toolkit or device.
static_assert(has_data<gpu::Vector &> && has_data<const gpu::Vector &>);
static_assert(!has_data<gpu::Vector> && !has_data<const gpu::Vector>);
static_assert(has_view<gpu::Vector &> && has_view<const gpu::Vector &>);
static_assert(!has_view<gpu::Vector> && !has_view<const gpu::Vector>);
static_assert(std::is_same_v<decltype(std::declval<gpu::Vector &>().data()), double *>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::Vector &>().data()), const double *>);
static_assert(std::is_same_v<decltype(std::declval<gpu::Vector &>().view()), gpu::VectorView>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::Vector &>().view()), gpu::ConstVectorView>);
static_assert(has_view<gpu::DenseMatrix &> && has_view<const gpu::DenseMatrix &>);
static_assert(!has_view<gpu::DenseMatrix> && !has_view<const gpu::DenseMatrix>);
static_assert(has_view<gpu::CsrMatrix &> && has_view<const gpu::CsrMatrix &>);
static_assert(!has_view<gpu::CsrMatrix> && !has_view<const gpu::CsrMatrix>);
static_assert(has_inverse_diagonal<gpu::JacobiPreconditioner &> &&
              has_inverse_diagonal<const gpu::JacobiPreconditioner &>);
static_assert(!has_inverse_diagonal<gpu::JacobiPreconditioner> &&
              !has_inverse_diagonal<const gpu::JacobiPreconditioner>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::JacobiPreconditioner &>().inverse_diagonal()),
                             gpu::ConstVectorView>);
static_assert(has_context<gpu::Workspace &> && has_context<const gpu::Workspace &>);
static_assert(!has_context<gpu::Workspace> && !has_context<const gpu::Workspace>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::Workspace &>().context()), const gpu::Context &>);
static_assert(has_residual<gpu::Workspace &> && has_direction<gpu::Workspace &> && has_product<gpu::Workspace &> &&
              has_scratch<gpu::Workspace &> && has_preconditioned<gpu::Workspace &>);
static_assert(!has_residual<gpu::Workspace> && !has_direction<gpu::Workspace> && !has_product<gpu::Workspace> &&
              !has_scratch<gpu::Workspace> && !has_preconditioned<gpu::Workspace>);
static_assert(!has_residual<const gpu::Workspace &> && !has_direction<const gpu::Workspace &> &&
              !has_product<const gpu::Workspace &> && !has_scratch<const gpu::Workspace &> &&
              !has_preconditioned<const gpu::Workspace &>);
static_assert(!has_residual<const gpu::Workspace> && !has_direction<const gpu::Workspace> &&
              !has_product<const gpu::Workspace> && !has_scratch<const gpu::Workspace> &&
              !has_preconditioned<const gpu::Workspace>);
// Temporary descriptors remain valid because they do not own their pointees.
static_assert(has_data<gpu::VectorView> && has_data<const gpu::VectorView>);
static_assert(has_data<gpu::ConstVectorView> && has_data<const gpu::ConstVectorView>);
static_assert(has_data<gpu::DenseMatrixView> && has_data<const gpu::DenseMatrixView>);
static_assert(has_data<gpu::CsrMatrixView> && has_data<const gpu::CsrMatrixView>);

template <class View>
constexpr bool lightweight_descriptor = std::is_trivially_copyable_v<View> && std::is_standard_layout_v<View> &&
                                        std::is_nothrow_default_constructible_v<View> && !std::is_aggregate_v<View>;

static_assert(lightweight_descriptor<gpu::ConstVectorView>);
static_assert(lightweight_descriptor<gpu::VectorView>);
static_assert(lightweight_descriptor<gpu::DenseMatrixView>);
static_assert(lightweight_descriptor<gpu::CsrMatrixView>);
static_assert(std::is_convertible_v<gpu::VectorView, gpu::ConstVectorView>);
static_assert(!std::is_convertible_v<gpu::ConstVectorView, gpu::VectorView>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::VectorView>().data()), double *>);
static_assert(std::is_same_v<decltype(std::declval<const gpu::ConstVectorView>().data()), const double *>);

constexpr gpu::VectorView empty_vector;
constexpr gpu::ConstVectorView empty_const_vector = empty_vector;
constexpr gpu::DenseMatrixView empty_dense;
constexpr gpu::CsrMatrixView empty_csr;
static_assert(empty_vector.data() == nullptr && empty_vector.size() == 0 && empty_vector.device() == -1);
static_assert(empty_const_vector.data() == nullptr && empty_const_vector.size() == 0 &&
              empty_const_vector.device() == -1);
static_assert(empty_dense.data() == nullptr && empty_dense.rows() == 0 && empty_dense.cols() == 0 &&
              empty_dense.row_stride() == 0 && empty_dense.device() == -1);
static_assert(empty_csr.indptr() == nullptr && empty_csr.indices() == nullptr && empty_csr.data() == nullptr &&
              empty_csr.rows() == 0 && empty_csr.cols() == 0 && empty_csr.nnz() == 0 &&
              empty_csr.index_type() == gpu::IndexType::int64 && empty_csr.device() == -1);

// Construction packages metadata only, including descriptors that a consuming
// boundary must reject. It must remain valid in constant evaluation.
constexpr gpu::VectorView unchecked_vector(nullptr, 3, 7);
constexpr gpu::ConstVectorView unchecked_const_vector = unchecked_vector;
constexpr gpu::DenseMatrixView unchecked_dense(nullptr, 3, 5, 2, 7);
constexpr gpu::CsrMatrixView unchecked_csr(nullptr, nullptr, nullptr, 3, 5, 8, gpu::IndexType::int32, 7);
static_assert(unchecked_const_vector.size() == 3 && unchecked_const_vector.device() == 7);
static_assert(unchecked_dense.rows() == 3 && unchecked_dense.cols() == 5 && unchecked_dense.row_stride() == 2 &&
              unchecked_dense.device() == 7);
static_assert(unchecked_csr.rows() == 3 && unchecked_csr.cols() == 5 && unchecked_csr.nnz() == 8 &&
              unchecked_csr.index_type() == gpu::IndexType::int32 && unchecked_csr.device() == 7);
static_assert(noexcept(gpu::VectorView(nullptr, 3, 7)));
static_assert(noexcept(gpu::ConstVectorView(nullptr, 3, 7)));
static_assert(noexcept(gpu::DenseMatrixView(nullptr, 3, 5, 2, 7)));
static_assert(noexcept(gpu::CsrMatrixView(nullptr, nullptr, nullptr, 3, 5, 8, gpu::IndexType::int32, 7)));
static_assert(noexcept(static_cast<gpu::ConstVectorView>(empty_vector)));

TEST(CudaViews, VectorCopiesAndConstConversionPreserveBorrowedMetadata)
{
    double storage[3]{};
    const gpu::VectorView mutable_view(storage, 3, 2);
    const auto copy = mutable_view;
    const gpu::ConstVectorView const_view = copy;
    EXPECT_EQ(copy.data(), storage);
    EXPECT_EQ(const_view.data(), storage);
    EXPECT_EQ(const_view.size(), 3u);
    EXPECT_EQ(const_view.device(), 2);
}

TEST(CudaViews, DenseMetadataPreservesPaddingAndDevice)
{
    const double storage[8]{};
    const gpu::DenseMatrixView view(storage, 2, 3, 5, 4);
    const auto copy = view;
    EXPECT_EQ(copy.data(), storage);
    EXPECT_EQ(copy.rows(), 2u);
    EXPECT_EQ(copy.cols(), 3u);
    EXPECT_EQ(copy.row_stride(), 5u);
    EXPECT_EQ(copy.device(), 4);
}

TEST(CudaViews, CsrMetadataPreservesIndexWidthAndAllBorrowedPointers)
{
    const std::int32_t indptr32[]{0, 1, 2}, indices32[]{0, 1};
    const std::int64_t indptr64[]{0, 1, 2}, indices64[]{0, 1};
    const double values[]{2.0, 3.0};
    const gpu::CsrMatrixView views[]{{indptr32, indices32, values, 2, 3, 2, gpu::IndexType::int32, 5},
                                     {indptr64, indices64, values, 2, 3, 2, gpu::IndexType::int64, 5}};
    const void *indptrs[]{indptr32, indptr64};
    const void *indices[]{indices32, indices64};
    for (std::size_t i = 0; i < 2; ++i)
    {
        const auto copy = views[i];
        EXPECT_EQ(copy.indptr(), indptrs[i]);
        EXPECT_EQ(copy.indices(), indices[i]);
        EXPECT_EQ(copy.data(), values);
        EXPECT_EQ(copy.rows(), 2u);
        EXPECT_EQ(copy.cols(), 3u);
        EXPECT_EQ(copy.nnz(), 2u);
        EXPECT_EQ(copy.index_type(), i == 0 ? gpu::IndexType::int32 : gpu::IndexType::int64);
        EXPECT_EQ(copy.device(), 5);
    }
}

} // namespace
