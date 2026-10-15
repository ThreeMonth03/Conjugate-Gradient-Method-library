#pragma once

#include <cg/matrix/cuda/csr_matrix.hpp>
#include <cg/matrix/cuda/dense_matrix.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/preconditioner/kind.hpp>
#include <cg/solver/cuda/conjugate_gradient.hpp>

#include <cstdint>
#include <optional>
#include <variant>

namespace cg
{
namespace cuda
{

// An immutable owning matrix and optional Jacobi inverse. Workspaces retain
// execution resources independently so one plan can serve different streams.
class PreparedSolver
{
    public:
    PreparedSolver(DenseMatrix matrix, const Context &context, DenseKernel kernel = DenseKernel::native,
                   PreconditionerKind preconditioner = PreconditionerKind::none);
    PreparedSolver(CsrMatrix matrix, const Context &context, DenseKernel kernel = DenseKernel::native,
                   PreconditionerKind preconditioner = PreconditionerKind::none);
    PreparedSolver(const PreparedSolver &) = delete;
    PreparedSolver &operator=(const PreparedSolver &) = delete;
    PreparedSolver(PreparedSolver &&) noexcept = default;
    PreparedSolver &operator=(PreparedSolver &&) noexcept = default;
    std::size_t size() const noexcept;
    int device() const noexcept { return m_device; }
    DenseKernel dense_kernel() const noexcept { return m_kernel; }
    PreconditionerKind preconditioner() const noexcept { return m_preconditioner_kind; }
    std::uint64_t preconditioner_setup_ns() const noexcept { return m_preconditioner_setup_ns; }
    std::size_t preconditioner_nbytes() const noexcept;
    Workspace create_workspace(std::uintptr_t stream = 0) const;
    SolveInfo solve_into(ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options = {},
                         bool fused = true) const;

    private:
    void prepare(const Context &context);
    std::variant<DenseMatrix, CsrMatrix> m_matrix;
    int m_device;
    DenseKernel m_kernel;
    PreconditionerKind m_preconditioner_kind;
    std::optional<JacobiPreconditioner> m_preconditioner;
    std::uint64_t m_preconditioner_setup_ns = 0;
};

} // namespace cuda
} // namespace cg
