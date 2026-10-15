#include <cg/matrix/validation.hpp>
#include <cg/solver/conjugate_gradient.hpp>
#include <cg/solver/detail/engine.hpp>

#include <cg/matrix/csr_matrix.hpp>
#include <cg/preconditioner/jacobi.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace cg
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
    if (rows(A) == 0 || rows(A) != cols(A) || b.size != rows(A) || x.size != rows(A))
    {
        throw std::invalid_argument("CG requires a nonempty square matrix and matching vectors");
    }
    if (workspace.size() != rows(A))
    {
        throw std::invalid_argument("prepare the workspace for the matrix dimension before solving");
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
        if (M->size() != rows(A))
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

} // namespace

const char *status_name(Status status) noexcept
{
    switch (status)
    {
    case Status::converged:
        return "converged";
    case Status::max_iterations:
        return "max_iterations";
    case Status::non_positive_curvature:
        return "non_positive_curvature";
    case Status::non_finite:
        return "non_finite";
    }
    return "unknown";
}

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

} // namespace cg
