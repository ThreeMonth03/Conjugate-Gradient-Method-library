#include <cg/backend/cuda/detail/backend.hpp>
#include <cg/matrix/cuda/detail/validation.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/solver/cuda/conjugate_gradient.hpp>
#include <cg/solver/cuda/detail/workspace_use_guard.hpp>
#include <cg/solver/detail/engine.hpp>
#include <cg/solver/detail/validation.hpp>

namespace cg
{
namespace cuda
{
namespace
{

std::size_t rows(DenseMatrixView A)
{
    return A.rows;
}
std::size_t rows(const CsrMatrix &A)
{
    return A.rows();
}
std::size_t cols(DenseMatrixView A)
{
    return A.cols;
}
std::size_t cols(const CsrMatrix &A)
{
    return A.cols();
}
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
void validate_solve(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                    const CgOptions &options, const JacobiPreconditioner *M)
{
    validate_matrix(A, context);
    detail::validate(b, context);
    detail::validate(x, context);
    if (rows(A) == 0 || rows(A) != cols(A) || b.size != rows(A) || x.size != rows(A))
    {
        throw std::invalid_argument("CUDA CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != rows(A) || workspace.device() != context.device() || workspace.stream() != context.stream())
    {
        throw std::invalid_argument("CUDA workspace must match the system size, device and stream");
    }
    cg::detail::validate_options(options, M != nullptr);
    if (M != nullptr)
    {
        if (M->size() != rows(A) || M->device() != context.device())
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
SolveInfo run(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
              const CgOptions &options, bool fused, DenseKernel kernel, const JacobiPreconditioner *M)
{
    validate_solve(A, b, x, workspace, context, options, M);
    detail::WorkspaceUseGuard use(workspace, context);
    detail::DeviceGuard guard(context.device());
    const detail::Backend backend(use.resources(), fused, kernel);
    const auto norm = [&](ConstVectorView value, double squared) { return backend.norm_from_squared(value, squared); };
    const auto info = options.algorithm == Algorithm::explicit_residual
                          ? cg::detail::solve_explicit(A, b, x, workspace, backend, options, norm)
                          : cg::detail::solve_recurrence(A, b, x, workspace, backend, options, M, norm);
    context.synchronize();
    use.complete();
    return info;
}

template <class Matrix>
SolveInfo solve_with_preconditioner(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                                    const Context &context, const CgOptions &options, bool fused, DenseKernel kernel,
                                    PreconditionerKind preconditioner)
{
    if (preconditioner != PreconditionerKind::none && preconditioner != PreconditionerKind::jacobi)
    {
        throw std::invalid_argument("unsupported preconditioner kind");
    }
    cg::detail::validate_options(options, preconditioner == PreconditionerKind::jacobi);
    if (preconditioner == PreconditionerKind::none)
    {
        return run(A, b, x, workspace, context, options, fused, kernel, nullptr);
    }
    if (preconditioner == PreconditionerKind::jacobi)
    {
        const JacobiPreconditioner M(A, context);
        return run(A, b, x, workspace, context, options, fused, kernel, &M);
    }
    throw std::invalid_argument("unsupported preconditioner kind");
}

} // namespace

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, const JacobiPreconditioner *M)
{
    return run(A, b, x, workspace, context, options, fused, kernel, M);
}
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, const JacobiPreconditioner *M)
{
    if (kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    return run(A, b, x, workspace, context, options, fused, kernel, M);
}

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, PreconditionerKind preconditioner)
{
    return solve_with_preconditioner(A, b, x, workspace, context, options, fused, kernel, preconditioner);
}
SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
                     const CgOptions &options, bool fused, DenseKernel kernel, PreconditionerKind preconditioner)
{
    if (kernel != DenseKernel::native)
    {
        throw std::invalid_argument("CUDA cuBLAS dense kernels do not support CSR");
    }
    return solve_with_preconditioner(A, b, x, workspace, context, options, fused, kernel, preconditioner);
}

} // namespace cuda
} // namespace cg
