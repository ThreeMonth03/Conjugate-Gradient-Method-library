#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/matrix/cuda/detail/validation.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/runtime/cuda/detail/runtime.hpp>
#include <cg/solver/cuda/conjugate_gradient.hpp>
#include <cg/solver/cuda/detail/device_execution.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>
#include <cg/solver/detail/engine.hpp>
#include <cg/solver/detail/validation.hpp>

namespace cg
{
namespace cuda
{
namespace
{

void validate_matrix(DenseMatrixView A, const Context &context)
{
    detail::validate(A, context);
}
void validate_matrix(const CsrMatrix &A, const Context &context)
{
    if (A.device() != context.device())
    {
        throw std::invalid_argument("CUDA CSR matrix belongs to a different device");
    }
}

template <class Matrix>
void validate_solve(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                    const JacobiPreconditioner *M)
{
    const Context &context = workspace.context();
    validate_matrix(A, context);
    detail::validate(b, context);
    detail::validate(x, context);
    if (A.rows() == 0 || A.rows() != A.cols() || b.size() != A.rows() || x.size() != A.rows())
    {
        throw std::invalid_argument("CUDA CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != A.rows())
    {
        throw std::invalid_argument("CUDA workspace must match the system size");
    }
    cg::detail::validate_options(options, M != nullptr);
    if (M != nullptr)
    {
        if (M->size() != A.rows() || M->device() != context.device())
        {
            throw std::invalid_argument("CUDA Jacobi dimension or device does not match");
        }
        if (detail::overlaps(M->inverse_diagonal(), x))
        {
            throw std::invalid_argument("CUDA output and preconditioner storage overlap");
        }
    }
    if (detail::overlaps(b, x) || detail::overlaps(A, x))
    {
        throw std::invalid_argument("CUDA CG output must be disjoint from inputs");
    }
    const ConstVectorView vectors[] = {workspace.residual(), workspace.direction(), workspace.product(),
                                       workspace.scratch(), workspace.preconditioned()};
    for (const auto vector : vectors)
    {
        if (detail::overlaps(vector, x) || detail::overlaps(vector, b) || detail::overlaps(A, vector) ||
            (M != nullptr && detail::overlaps(M->inverse_diagonal(), vector)))
        {
            throw std::invalid_argument("CUDA workspace must be disjoint from inputs and output");
        }
    }
}

template <class Matrix>
SolveInfo run(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
              bool fused, const JacobiPreconditioner *M)
{
    const Context &context = workspace.context();
    validate_solve(A, b, x, workspace, options, M);
    detail::WorkspaceUseGuard use(workspace);
    detail::DeviceGuard guard(context.device());
    const detail::Backend backend(use.resources(), fused);
    const auto norm = [&](ConstVectorView value, double squared) { return backend.norm_from_squared(value, squared); };
    SolveInfo info;
    if (workspace.execution_mode() == ExecutionMode::device)
    {
        info = detail::solve_device(A, b, x, workspace, backend, use.resources(), use.device_resources(), options,
                                    fused, M);
    }
    else
    {
        info = options.algorithm == Algorithm::explicit_residual
                   ? cg::detail::solve_explicit(A, b, x, workspace, backend, options, norm)
                   : cg::detail::solve_recurrence(A, b, x, workspace, backend, options, M, norm);
    }
    context.synchronize();
    use.complete();
    return info;
}

template <class Matrix>
SolveInfo solve_with_preconditioner(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                                    const CgOptions &options, bool fused, PreconditionerKind preconditioner)
{
    if (preconditioner != PreconditionerKind::none && preconditioner != PreconditionerKind::jacobi)
    {
        throw std::invalid_argument("unsupported preconditioner kind");
    }
    cg::detail::validate_options(options, preconditioner == PreconditionerKind::jacobi);
    if (preconditioner == PreconditionerKind::none)
    {
        return run(A, b, x, workspace, options, fused, nullptr);
    }
    if (preconditioner == PreconditionerKind::jacobi)
    {
        const JacobiPreconditioner M(A, workspace.context());
        return run(A, b, x, workspace, options, fused, &M);
    }
    throw std::invalid_argument("unsupported preconditioner kind");
}

} // namespace

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                     bool fused, const JacobiPreconditioner *M)
{
    return run(A, b, x, workspace, options, fused, M);
}
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CgOptions &options, bool fused, const JacobiPreconditioner *M)
{
    if (workspace.dense_kernel() != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    return run(A, b, x, workspace, options, fused, M);
}

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                     bool fused, PreconditionerKind preconditioner)
{
    return solve_with_preconditioner(A, b, x, workspace, options, fused, preconditioner);
}
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CgOptions &options, bool fused, PreconditionerKind preconditioner)
{
    if (workspace.dense_kernel() != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    return solve_with_preconditioner(A, b, x, workspace, options, fused, preconditioner);
}

} // namespace cuda
} // namespace cg
