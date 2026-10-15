#include <cg/solver/prepared_solver.hpp>

#include <chrono>
#include <stdexcept>
#include <utility>

namespace cg
{
namespace
{

DenseMatrixView operator_view(const DenseMatrix &matrix)
{
    return matrix.view();
}
const CsrMatrix &operator_view(const CsrMatrix &matrix)
{
    return matrix;
}

} // namespace

PreparedSolver::PreparedSolver(DenseMatrix matrix, int num_threads, PreconditionerKind preconditioner,
                               DenseKernel kernel)
    : PreparedSolver(Matrix(std::move(matrix)), num_threads, preconditioner, kernel)
{
}
PreparedSolver::PreparedSolver(CsrMatrix matrix, int num_threads, PreconditionerKind preconditioner, DenseKernel kernel)
    : PreparedSolver(Matrix(std::move(matrix)), num_threads, preconditioner, kernel)
{
}
PreparedSolver::PreparedSolver(Matrix matrix, int num_threads, PreconditionerKind preconditioner, DenseKernel kernel)
    : m_matrix(std::move(matrix)), m_num_threads(num_threads), m_kernel(kernel)
{
    if (preconditioner != PreconditionerKind::none && preconditioner != PreconditionerKind::jacobi)
    {
        throw std::invalid_argument("unsupported preconditioner");
    }
    if (std::holds_alternative<CsrMatrix>(m_matrix) && kernel != DenseKernel::native)
    {
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    }
    m_num_threads = CpuBackend(num_threads, true, kernel).threads();
    std::visit(
        [](const auto &matrix)
        {
            if (matrix.rows() == 0 || matrix.rows() != matrix.cols())
            {
                throw std::invalid_argument("CG requires a nonempty square matrix");
            }
        },
        m_matrix);
    if (preconditioner == PreconditionerKind::jacobi)
    {
        const auto start = std::chrono::steady_clock::now();
        std::visit([this](const auto &matrix) { m_preconditioner.emplace(operator_view(matrix)); }, m_matrix);
        m_setup_ns =
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
    }
}

std::size_t PreparedSolver::size() const noexcept
{
    return std::visit([](const auto &matrix) { return matrix.rows(); }, m_matrix);
}
PreconditionerKind PreparedSolver::preconditioner() const noexcept
{
    return m_preconditioner ? PreconditionerKind::jacobi : PreconditionerKind::none;
}
std::size_t PreparedSolver::preconditioner_nbytes() const noexcept
{
    return m_preconditioner ? m_preconditioner->inverse_diagonal().size_bytes() : 0;
}
Workspace PreparedSolver::create_workspace() const
{
    return Workspace(size());
}
SolveInfo PreparedSolver::solve_into(ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                                     bool fused) const
{
    const CpuBackend backend(m_num_threads, fused, m_kernel);
    const auto *preconditioner = m_preconditioner ? &*m_preconditioner : nullptr;
    return std::visit(
        [&](const auto &matrix)
        { return cg::solve_into(operator_view(matrix), b, x, workspace, backend, options, preconditioner); }, m_matrix);
}

} // namespace cg
