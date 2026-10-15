#include <cg/matrix/cuda/validation.hpp>
#include <cg/preconditioner/cuda/jacobi.hpp>
#include <cg/solver/cuda/conjugate_gradient.hpp>
#include <cg/solver/detail/engine.hpp>

#include <cmath>

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
    if (!std::isfinite(options.rtol) || !std::isfinite(options.atol) || options.rtol < 0.0 || options.atol < 0.0)
    {
        throw std::invalid_argument("tolerances must be finite and nonnegative");
    }
    if (options.algorithm != Algorithm::explicit_residual && options.algorithm != Algorithm::recurrence)
    {
        throw std::invalid_argument("unsupported CG algorithm");
    }
    if (M != nullptr)
    {
        if (options.algorithm != Algorithm::recurrence)
        {
            throw std::invalid_argument("Jacobi preconditioning requires recurrence CG");
        }
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

struct WorkspaceUse
{
    WorkspaceUse(Workspace &workspace, const Context &context) : workspace(workspace), context(context)
    {
        if (!workspace.try_acquire())
        {
            throw std::invalid_argument("CUDA workspace is already in use by another solve");
        }
    }
    WorkspaceUse(const WorkspaceUse &) = delete;
    WorkspaceUse &operator=(const WorkspaceUse &) = delete;
    ~WorkspaceUse()
    {
        if (!completed)
        {
            try
            {
                context.synchronize();
            }
            catch (...)
            {
            }
        }
        workspace.release();
    }
    Workspace &workspace;
    const Context &context;
    bool completed = false;
};

template <class Matrix>
SolveInfo run(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const Context &context,
              const CgOptions &options, bool fused, DenseKernel kernel, const JacobiPreconditioner *M)
{
    validate_solve(A, b, x, workspace, context, options, M);
    WorkspaceUse use(workspace, context);
    detail::DeviceGuard guard(context.device());
    const Backend backend(context, workspace, fused, kernel);
    const auto norm = [&](ConstVectorView value, double squared) { return backend.norm_from_squared(value, squared); };
    const auto info = options.algorithm == Algorithm::explicit_residual
                          ? cg::detail::solve_explicit(A, b, x, workspace, backend, options, norm)
                          : cg::detail::solve_recurrence(A, b, x, workspace, backend, options, M, norm);
    context.synchronize();
    use.completed = true;
    return info;
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

} // namespace cuda
} // namespace cg
