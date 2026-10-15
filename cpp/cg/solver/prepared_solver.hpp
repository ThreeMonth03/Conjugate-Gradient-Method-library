#pragma once

#include <cg/matrix/csr_matrix.hpp>
#include <cg/matrix/dense_matrix.hpp>
#include <cg/preconditioner/jacobi.hpp>
#include <cg/preconditioner/kind.hpp>
#include <cg/solver/conjugate_gradient.hpp>

#include <cstdint>
#include <optional>
#include <variant>

namespace cg
{

// Immutable operator snapshot and prepared state. Each concurrent solve needs
// its own workspace/output; tolerances and fusion remain per-call choices.
class PreparedSolver
{
    public:
    PreparedSolver(DenseMatrix matrix, int num_threads = 1,
                   PreconditionerKind preconditioner = PreconditionerKind::none,
                   DenseKernel kernel = DenseKernel::native);
    PreparedSolver(CsrMatrix matrix, int num_threads = 1, PreconditionerKind preconditioner = PreconditionerKind::none,
                   DenseKernel kernel = DenseKernel::native);
    PreparedSolver(const PreparedSolver &) = delete;
    PreparedSolver &operator=(const PreparedSolver &) = delete;
    PreparedSolver(PreparedSolver &&) noexcept = default;
    PreparedSolver &operator=(PreparedSolver &&) noexcept = default;

    std::size_t size() const noexcept;
    int num_threads() const noexcept { return m_num_threads; }
    DenseKernel dense_kernel() const noexcept { return m_kernel; }
    PreconditionerKind preconditioner() const noexcept;
    std::uint64_t preconditioner_setup_ns() const noexcept { return m_setup_ns; }
    std::size_t preconditioner_nbytes() const noexcept;
    Workspace create_workspace() const;
    SolveInfo solve_into(ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options = {},
                         bool fused = true) const;

    private:
    using Matrix = std::variant<DenseMatrix, CsrMatrix>;
    PreparedSolver(Matrix matrix, int num_threads, PreconditionerKind preconditioner, DenseKernel kernel);

    Matrix m_matrix;
    int m_num_threads;
    DenseKernel m_kernel;
    std::optional<JacobiPreconditioner> m_preconditioner;
    std::uint64_t m_setup_ns = 0;
};

} // namespace cg
