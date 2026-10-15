#include <cg/backend/cpu/detail/reduction.hpp>
#include <cg/matrix/detail/validation.hpp>
#include <cg/solver/conjugate_gradient.hpp>
#include <cg/solver/detail/engine.hpp>
#include <cg/solver/detail/validation.hpp>

#include <cg/matrix/csr_matrix.hpp>
#include <cg/preconditioner/jacobi.hpp>

#include <stdexcept>

namespace cg
{
namespace
{

void validate_matrix(DenseMatrixView A)
{
    detail::validate(A);
}
void validate_matrix(const CsrMatrix &) {} // Validated, immutable owning storage.

template <class Matrix>
void validate_solve(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace, const CgOptions &options,
                    const JacobiPreconditioner *M)
{
    validate_matrix(A);
    detail::validate(b);
    detail::validate(x);
    if (A.rows() == 0 || A.rows() != A.cols() || b.size() != A.rows() || x.size() != A.rows())
    {
        throw std::invalid_argument("CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != A.rows())
    {
        throw std::invalid_argument("prepare the workspace for the matrix dimension before solving");
    }
    detail::validate_options(options, M != nullptr);
    if (M != nullptr)
    {
        if (M->size() != A.rows())
        {
            throw std::invalid_argument("Jacobi preconditioner dimension does not match the matrix");
        }
        if (detail::overlaps(M->inverse_diagonal(), x))
        {
            throw std::invalid_argument("CG output must be disjoint from the preconditioner");
        }
    }
    if (detail::overlaps(b, x) || detail::overlaps(A, x))
    {
        throw std::invalid_argument("CG output must be disjoint from inputs");
    }
    const ConstVectorView vectors[] = {workspace.residual(), workspace.direction(), workspace.product(),
                                       workspace.scratch(), workspace.preconditioned()};
    for (const auto vector : vectors)
    {
        if (detail::overlaps(vector, x) || detail::overlaps(vector, b) || detail::overlaps(A, vector) ||
            (M != nullptr && detail::overlaps(M->inverse_diagonal(), vector)))
        {
            throw std::invalid_argument("workspace must be disjoint from CG inputs and output");
        }
    }
}

template <class Matrix>
SolveInfo solve_with_setup(const Matrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                           const CpuBackend &backend, const CgOptions &options, PreconditionerKind preconditioner)
{
    if (preconditioner == PreconditionerKind::none)
    {
        return solve_into(A, b, x, workspace, backend, options);
    }
    if (preconditioner != PreconditionerKind::jacobi)
    {
        throw std::invalid_argument("unsupported preconditioner");
    }
    detail::validate_options(options, true);
    const JacobiPreconditioner prepared(A);
    return solve_into(A, b, x, workspace, backend, options, &prepared);
}

} // namespace

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, const JacobiPreconditioner *M)
{
    validate_solve(A, b, x, workspace, options, M);
    return options.algorithm == Algorithm::explicit_residual
               ? detail::solve_explicit(A, b, x, workspace, backend, options, detail::norm_from_squared)
               : detail::solve_recurrence(A, b, x, workspace, backend, options, M, detail::norm_from_squared);
}

SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, const JacobiPreconditioner *M)
{
    if (backend.dense_kernel() == DenseKernel::blas)
    {
        throw std::invalid_argument("BLAS dense kernels do not support CSR matrices");
    }
    validate_solve(A, b, x, workspace, options, M);
    return options.algorithm == Algorithm::explicit_residual
               ? detail::solve_explicit(A, b, x, workspace, backend, options, detail::norm_from_squared)
               : detail::solve_recurrence(A, b, x, workspace, backend, options, M, detail::norm_from_squared);
}

SolveInfo solve_into(DenseMatrixView A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, PreconditionerKind preconditioner)
{
    return solve_with_setup(A, b, x, workspace, backend, options, preconditioner);
}

SolveInfo solve_into(const CsrMatrix &A, ConstVectorView b, VectorView x, Workspace &workspace,
                     const CpuBackend &backend, const CgOptions &options, PreconditionerKind preconditioner)
{
    return solve_with_setup(A, b, x, workspace, backend, options, preconditioner);
}

} // namespace cg
