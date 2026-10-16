#include <cg/solver/cuda/prepared_solver.hpp>

#include <chrono>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace cg
{
namespace cuda
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

PreparedSolver::PreparedSolver(DenseMatrix matrix, const Context &context, PreconditionerKind preconditioner,
                               DenseKernel kernel)
    : m_matrix(std::move(matrix)), m_device(context.device()), m_kernel(kernel)
{
    prepare(context, preconditioner);
}
PreparedSolver::PreparedSolver(CsrMatrix matrix, const Context &context, PreconditionerKind preconditioner,
                               DenseKernel kernel)
    : m_matrix(std::move(matrix)), m_device(context.device()), m_kernel(kernel)
{
    prepare(context, preconditioner);
}
void PreparedSolver::prepare(const Context &context, PreconditionerKind preconditioner)
{
    if (m_kernel != DenseKernel::native && m_kernel != DenseKernel::cublas)
    {
        throw std::invalid_argument("unsupported CUDA dense kernel");
    }
    if (preconditioner != PreconditionerKind::none && preconditioner != PreconditionerKind::jacobi)
    {
        throw std::invalid_argument("unsupported preconditioner kind");
    }
    std::visit(
        [&](const auto &matrix)
        {
            if (matrix.device() != context.device())
            {
                throw std::invalid_argument("CUDA matrix belongs to a different device");
            }
            if (matrix.rows() == 0 || matrix.rows() != matrix.cols())
            {
                throw std::invalid_argument("CG requires a nonempty square matrix");
            }
            if constexpr (std::is_same_v<std::decay_t<decltype(matrix)>, CsrMatrix>)
            {
                if (m_kernel != DenseKernel::native)
                {
                    throw std::invalid_argument("CUDA CSR requires dense_kernel='native'");
                }
            }
            if (preconditioner == PreconditionerKind::jacobi)
            {
                const auto start = std::chrono::steady_clock::now();
                m_preconditioner.emplace(operator_view(matrix), context);
                m_preconditioner_setup_ns =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start)
                        .count();
            }
        },
        m_matrix);
}
std::size_t PreparedSolver::size() const noexcept
{
    return std::visit([](const auto &matrix) { return matrix.rows(); }, m_matrix);
}
std::size_t PreparedSolver::preconditioner_nbytes() const noexcept
{
    return m_preconditioner ? size() * sizeof(double) : 0;
}
Workspace PreparedSolver::create_workspace(std::uintptr_t stream, ExecutionMode mode) const
{
    return Workspace(size(), Context(m_device, stream), m_kernel, mode);
}
SolveInfo PreparedSolver::solve_into(ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                                     bool fused) const
{
    if (workspace.device() != m_device)
    {
        throw std::invalid_argument("workspace device does not match prepared solver");
    }
    if (workspace.dense_kernel() != m_kernel)
    {
        throw std::invalid_argument("workspace dense kernel does not match prepared solver");
    }
    return std::visit(
        [&](const auto &matrix)
        {
            return cg::cuda::solve_into(operator_view(matrix), b, x, workspace, options, fused,
                                        m_preconditioner ? &*m_preconditioner : nullptr);
        },
        m_matrix);
}

} // namespace cuda
} // namespace cg
